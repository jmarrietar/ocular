# ocular

Description


# Train

#### SimCLR - TPU

```bash
python utils/download.py -d train_voets
```

```bash
python train/simclr/pre_train_simclr.py --data_dir train_voets --num_epochs 500 --resume-epochs 400  --num_workers 8
```

```bash
python train/simclr/pre_train_simclr.py --data_dir train_voets --num_epochs 300
```
