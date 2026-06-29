
## LET-NET Training Code 


## Requirements

- torch 2.0.1
- pytorch-lightning 2.07
- opencv-python 4.8
- tensorboard 2.13.0

## Datasets

- HPatches
- megadepth
- imw2020-val

Download this datasets and change path value in `training/train.py(line 121)`

## Training

run:

``` python
python training/train.py
```

tensorboard visualization

```shell
tensorboard --logdir=./log_train/
```

## Convert model

run:

``` python
python nets/alnet.py
```

Convert the trained model to a minimal model.

## Thanks

ALIKE training code https://github.com/Shiaoming/ALIKE

