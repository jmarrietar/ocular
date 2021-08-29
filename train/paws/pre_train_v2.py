import argparse
import yaml
import pprint
import logging
import os


logging.basicConfig()
logger = logging.getLogger()

crop_scale = (0.14, 1.0) if multicrop > 0 else (0.08, 1.0)
mc_scale = (0.05, 0.14)
mc_size = 96


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname", type=str, help="name of config file to load", default="configs.yaml"
    )
    parser.add_argument("--batch-size", default=64, type=int, metavar="N")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--learning-rate", default=0.00001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--num_cores",
        default=8,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--log_steps", default=20, type=int, help="Log every n steps")
    parser.add_argument("--metrics_debug", default=False, type=bool)
    parser.add_argument("--resume-epochs", type=int)
    parser.add_argument("--save-drive", default=False, action="store_true")

    return parser


def init_model(device, model_name="resnet50", use_pred=False, output_dim=128):
    if "wide_resnet" in model_name:
        encoder = wide_resnet.__dict__[model_name](dropout_rate=0.0)
        hidden_dim = 128
    else:
        encoder = resnet.__dict__[model_name]()

        # Load pre-trained ResNetImagenNet
        logger.info("Load pre-trained ResNet ImagenNet weigths ...")
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth", progress=True
        )
        log = encoder.load_state_dict(state_dict, strict=False)
        logger.info(log)

        hidden_dim = 2048
        if "w2" in model_name:
            hidden_dim *= 2
        elif "w4" in model_name:
            hidden_dim *= 4

    # -- projection head
    encoder.fc = torch.nn.Sequential(
        OrderedDict(
            [
                ("fc1", torch.nn.Linear(hidden_dim, hidden_dim)),
                ("bn1", torch.nn.BatchNorm1d(hidden_dim)),
                ("relu1", torch.nn.ReLU(inplace=True)),
                ("fc2", torch.nn.Linear(hidden_dim, hidden_dim)),
                ("bn2", torch.nn.BatchNorm1d(hidden_dim)),
                ("relu2", torch.nn.ReLU(inplace=True)),
                ("fc3", torch.nn.Linear(hidden_dim, output_dim)),
            ]
        )
    )

    # -- prediction head
    encoder.pred = None
    if use_pred:
        mx = 4  # 4x bottleneck prediction head
        pred_head = OrderedDict([])
        pred_head["bn1"] = torch.nn.BatchNorm1d(output_dim)
        pred_head["fc1"] = torch.nn.Linear(output_dim, output_dim // mx)
        pred_head["bn2"] = torch.nn.BatchNorm1d(output_dim // mx)
        pred_head["relu"] = torch.nn.ReLU(inplace=True)
        pred_head["fc2"] = torch.nn.Linear(output_dim // mx, output_dim)
        encoder.pred = torch.nn.Sequential(pred_head)

    # encoder.to(device)
    logger.info(encoder)
    return encoder


def sharpen_func(p):
    sharp_p = p ** (1.0 / T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p


def snn(device, query, supports, labels, tau):

    softmax = torch.nn.Softmax(dim=1)
    """ Soft Nearest Neighbours similarity classifier """
    # Step 1: normalize embeddings
    query = torch.nn.functional.normalize(query)
    supports = torch.nn.functional.normalize(supports)

    # Step 2: gather embeddings from all workers
    supports = AllGather.apply(supports)

    # Step 3: compute similarlity between local embeddings
    result = softmax(query @ supports.T / tau) @ labels
    # result = torch.matmul(softmax(torch.matmul(query, supports.T / tau)), labels)

    return result


def my_loss_func(
    device,
    anchor_views,
    anchor_supports,
    anchor_support_labels,
    target_views,
    target_supports,
    target_support_labels,
    sharpen=sharpen_func,
    snn=snn,
):
    # -- NOTE: num views of each unlabeled instance = 2+multicrop
    batch_size = len(anchor_views) // (2 + multicrop)

    # Step 1: compute anchor predictions
    probs = snn(device, anchor_views, anchor_supports, anchor_support_labels, tau)

    # Step 2: compute targets for anchor predictions
    with torch.no_grad():
        targets = snn(device, target_views, target_supports, target_support_labels, tau)
        targets = sharpen(targets)
        if multicrop > 0:
            mc_target = 0.5 * (targets[:batch_size] + targets[batch_size:])
            targets = torch.cat(
                [targets, *[mc_target for _ in range(multicrop)]], dim=0
            )
        targets[targets < 1e-4] *= 0  # numerical stability

    # Step 3: compute cross-entropy loss H(targets, queries)
    # loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))
    criterion = torch.nn.CrossEntropyLoss()
    targets2 = targets.argmax(-1)

    loss = criterion(probs, targets2)

    # Step 4: compute me-max regularizer
    rloss = 0.0
    if me_max:
        avg_probs = AllReduce.apply(torch.mean(sharpen(probs), dim=0))
        rloss -= torch.sum(torch.log(avg_probs ** (-avg_probs)))

    return loss, rloss


def train_resnet18():
    # -- load script params

    fname = FLAGS.fname

    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file)
        logger.info("loaded params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    model_name = params["meta"]["model_name"]
    output_dim = params["meta"]["output_dim"]
    multicrop = params["data"]["multicrop"]

    # -- CRITERTION
    reg = params["criterion"]["me_max"]
    supervised_views = params["criterion"]["supervised_views"]
    classes_per_batch = params["criterion"]["classes_per_batch"]
    s_batch_size = params["criterion"]["supervised_imgs_per_class"]
    u_batch_size = params["criterion"]["unsupervised_batch_size"]
    temperature = params["criterion"]["temperature"]
    sharpen = params["criterion"]["sharpen"]

    # -- DATA
    unlabeled_frac = params["data"]["unlabeled_frac"]
    color_jitter = params["data"]["color_jitter_strength"]
    normalize = params["data"]["normalize"]
    root_path = params["data"]["root_path"]
    s_image_folder = params["data"]["s_image_folder"]
    u_image_folder = params["data"]["u_image_folder"]
    dataset_name = params["data"]["dataset"]
    subset_path = params["data"]["subset_path"]
    unique_classes = params["data"]["unique_classes_per_rank"]
    label_smoothing = params["data"]["label_smoothing"]
    data_seed = None

    copy_data = params["meta"]["copy_data"]
    use_pred_head = params["meta"]["use_pred_head"]

    use_fp16 = params["meta"]["use_fp16"]

    # -- OPTIMIZATION
    wd = float(params["optimization"]["weight_decay"])
    num_epochs = params["optimization"]["epochs"]
    warmup = params["optimization"]["warmup"]
    start_lr = params["optimization"]["start_lr"]
    lr = params["optimization"]["lr"]
    final_lr = params["optimization"]["final_lr"]
    mom = params["optimization"]["momentum"]
    nesterov = params["optimization"]["nesterov"]

    # -- META
    load_model = params["meta"]["load_checkpoint"]
    r_file = params["meta"]["read_checkpoint"]

    # -- LOGGING
    folder = params["logging"]["folder"]
    tag = params["logging"]["write_tag"]

    torch.manual_seed(1)

    ############# PAWS CODE ##################
    device = xm.xla_device()

    # -- init model
    encoder = init_model(
        device=device,
        model_name=model_name,
        use_pred=use_pred_head,
        output_dim=output_dim,
    )

    encoder = encoder.to(device)  # YOO

    # -- make data transforms
    transform, init_transform = make_transforms(
        dataset_name=dataset_name,
        subset_path=subset_path,
        unlabeled_frac=unlabeled_frac,
        training=True,
        split_seed=data_seed,
        crop_scale=crop_scale,
        basic_augmentations=False,
        color_jitter=color_jitter,
        normalize=normalize,
    )

    multicrop_transform = (multicrop, None)
    if multicrop > 0:
        multicrop_transform = make_multicrop_transform(
            dataset_name=dataset_name,
            num_crops=multicrop,
            size=mc_size,
            crop_scale=mc_scale,
            normalize=normalize,
            color_distortion=color_jitter,
        )

    # -- init data-loaders/samplers
    (
        unsupervised_loader,
        unsupervised_sampler,
        supervised_loader,
        supervised_sampler,
    ) = init_data(
        dataset_name=dataset_name,
        transform=transform,
        init_transform=init_transform,
        supervised_views=supervised_views,
        u_batch_size=u_batch_size,
        s_batch_size=s_batch_size,
        unique_classes=unique_classes,
        classes_per_batch=classes_per_batch,
        multicrop_transform=multicrop_transform,
        world_size=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        root_path=root_path,
        s_image_folder=s_image_folder,
        u_image_folder=u_image_folder,
        training=True,
        copy_data=copy_data,
    )

    # iter_supervised = None
    ipe = len(unsupervised_loader)

    logger.info(f"iterations per epoch: {ipe}")

    # -- init optimizer and scheduler
    # scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # Scale learning rate to num cores
    learning_rate = FLAGS.get("learning_rate") * xm.xrt_world_size()

    # 2. TO DO: Optimizer Doble check learning rate
    optimizer = torch.optim.Adam(encoder.parameters(), learning_rate, weight_decay=5e-4)

    def train_loop_fn(supervised_loader, unsupervised_loader, epoch):

        tracker = xm.RateTracker()
        encoder.train()  # YOOO
        # -- TRAINING LOOP
        best_loss = None

        # unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        ploss_meter = AverageMeter()
        rloss_meter = AverageMeter()
        time_meter = AverageMeter()
        data_meter = AverageMeter()

        for itr, udata in enumerate(unsupervised_loader):

            def load_imgs(supervised_loader):
                # -- unsupervised imgs
                uimgs = [u.to(device, non_blocking=True) for u in udata[:-1]]

                # -- supervised imgs
                global iter_supervised
                try:
                    sdata = next(iter_supervised)
                except Exception:
                    iter_supervised = iter(supervised_loader)
                    sdata = next(iter_supervised)
                finally:
                    idx = sdata[1].clone().detach()
                    idx = idx.to(device)

                    labels_matrix = torch.zeros(
                        len(idx), idx.max() + 1, device=device
                    ).scatter_(1, idx.unsqueeze(1), 1.0)
                    labels_matrix = labels_matrix.to(device)  # YOOO

                    # Label Smoothing (mia y chambona)
                    labels_matrix = abs(labels_matrix - 0.05)  # YOOO

                    labels = torch.cat([labels_matrix for _ in range(supervised_views)])
                    simgs = [s.to(device, non_blocking=True) for s in sdata[:-1]]

                # -- concatenate supervised imgs and unsupervised imgs
                imgs = simgs + uimgs
                return imgs, labels

            imgs, labels = load_imgs(supervised_loader)

            def train_step():

                # with torch.cuda.amp.autocast(enabled=use_fp16):
                optimizer.zero_grad()

                # --
                # h: representations of 'imgs' before head
                # z: representations of 'imgs' after head
                # -- If use_pred_head=False, then encoder.pred (prediction
                #    head) is None, and _forward_head just returns the
                #    identity, z=h
                h, z = encoder(imgs, return_before_head=True)

                # Compute paws loss in full precision
                # with torch.cuda.amp.autocast(enabled=False):

                # Step 1. convert representations to fp32
                h, z = h.float(), z.float()

                # Step 2. determine anchor views/supports and their
                #         corresponding target views/supports
                # --
                num_support = supervised_views * s_batch_size * classes_per_batch

                # --
                anchor_supports = z[:num_support]
                anchor_views = z[num_support:]
                # --
                target_supports = h[:num_support].detach()
                target_views = h[num_support:].detach()
                target_views = torch.cat(
                    [
                        target_views[u_batch_size : 2 * u_batch_size],
                        target_views[:u_batch_size],
                    ],
                    dim=0,
                )

                # Step 3. compute paws loss with me-max regularization
                ploss, me_max = my_loss_func(
                    device=device,
                    anchor_views=anchor_views,
                    anchor_supports=anchor_supports,
                    anchor_support_labels=labels,
                    target_views=target_views,
                    target_supports=target_supports,
                    target_support_labels=labels,
                )

                loss = ploss + me_max
                loss.backward()

                xm.optimizer_step(optimizer)

                if itr % FLAGS.get("log_steps") == 0:
                    print(
                        "[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}".format(
                            xm.get_ordinal(),
                            itr,
                            loss.item(),
                            tracker.rate(),
                            tracker.global_rate(),
                            time.asctime(),
                        ),
                        flush=True,
                    )

                return (float(loss), float(ploss), float(me_max))

            (loss, ploss, rloss) = train_step()
            loss_meter.update(loss)
            ploss_meter.update(ploss)
            rloss_meter.update(rloss)

    data, pred, target = None, None, None

    start_epoch = 0
    end_epoch = FLAGS.get("num_epochs")

    train_supervised_loader = pl.MpDeviceLoader(supervised_loader, device)
    train_unsupervised_loader = pl.MpDeviceLoader(unsupervised_loader, device)

    for epoch in range(start_epoch, end_epoch):
        train_loop_fn(train_supervised_loader, train_unsupervised_loader, epoch)
        xm.master_print("Finished training epoch {}".format(epoch))
    # return accuracy, data, pred, target
    return 0, 0, 0, 0


def _mp_fn(rank, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type("torch.FloatTensor")
    accuracy, data, pred, target = train_resnet18()


if __name__ == "__main__":
    parser = init_argparse()
    global FLAGS
    FLAGS = parser.parse_args()
    print(FLAGS)
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS["num_cores"], start_method="fork")
