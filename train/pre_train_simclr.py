from utils.utils import save_checkpoint, accuracy_func
from utils.data_aug.gaussian_blur import GaussianBlur
from utils.data_aug.view_generator import (
    ContrastiveLearningViewGenerator,
    get_simclr_pipeline_transform,
)
from resnet_simclr import ResNetSimCLR
import torch
import torch.nn.functional as F
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torchvision
from torchvision import datasets, transforms
import logging
from torch.utils.tensorboard import SummaryWriter


def info_nce_loss(features, device):

    labels = torch.cat(
        [torch.arange(FLAGS["batch_size"]) for i in range(2)], dim=0
    )  # modifique a 2
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)

    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    TEMPERATURE = 0.07  # Yo lo Hardcodie

    logits = logits / TEMPERATURE
    return logits, labels


print("Hello Jose")


SERIAL_EXEC = xmp.MpSerialExecutor()

WRAPPED_MODEL = xmp.MpModelWrapper(ResNetSimCLR(base_model="resnet50"))

UNLABELED = "sample@1000"


def train_resnet():
    torch.manual_seed(1)

    def get_dataset():

        train_dataset = datasets.ImageFolder(
            root="{}".format(UNLABELED),
            transform=ContrastiveLearningViewGenerator(
                get_simclr_pipeline_transform(224), n_views=2
            ),
        )
        return train_dataset

    # Using the serial executor avoids multiple processes
    # to download the same data.
    train_dataset = SERIAL_EXEC.run(get_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS["batch_size"],
        sampler=train_sampler,
        num_workers=FLAGS["num_workers"],
        drop_last=True,
    )

    # Scale learning rate to num cores
    learning_rate = FLAGS["learning_rate"] * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss().to(device)  # YO

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()

        for x, (data, _) in enumerate(loader):
            optimizer.zero_grad()

            data = torch.cat(data, dim=0)

            output = model(data)
            logits, labels = info_nce_loss(output, device)  # YO

            loss = criterion(logits, labels)  # YO

            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS["batch_size"])

            top1, top5 = accuracy_func(logits, labels, topk=(1, 5))

            if x % FLAGS["log_steps"] == 0:
                print(
                    "[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}".format(
                        xm.get_ordinal(),
                        x,
                        loss.item(),
                        tracker.rate(),
                        tracker.global_rate(),
                        time.asctime(),
                    ),
                    flush=True,
                )
                print(f"Top1 accuracy: {top1[0]}")

    # Train and eval loops
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(1, FLAGS["num_epochs"] + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        xm.save(model.state_dict(), "net-DR-SimCLR.pt")

        if FLAGS["metrics_debug"]:
            xm.master_print(met.metrics_report(), flush=True)

    return accuracy, data, pred, target


# Start training processes
def _mp_fn(rank, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type("torch.FloatTensor")
    accuracy, data, pred, target = train_resnet()


xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS["num_cores"], start_method="fork")
