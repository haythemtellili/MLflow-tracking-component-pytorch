# Libraries
import mlflow
import tempfile
import os
import json
import logging
import argparse
import traceback
import warnings
from types import SimpleNamespace
import random
from pathlib import Path
import pandas as pd
import numpy as np

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from sklearn.model_selection import train_test_split

from train import train_cnn
from model import SimpleCNN
from dataset import get_augmentations, MyDataset

# Fix seed
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def split_data(df):
    # Splitting data into train and val and test
    train, test = train_test_split(
        df, stratify=df.label, test_size=0.1, random_state=42
    )
    train, val = train_test_split(
        train, stratify=train.label, test_size=0.15, random_state=42
    )
    return train, val, test


def init_data_loader(args):
    df = pd.read_csv(args.data_csv)
    train, val, test = split_data(df)
    
    tags1 = {
    f"Number of {e} for train": train[train["label"] == args.classes[e]].shape[0]
    for e in list(args.classes.keys())
    }
    tags2 = {
    f"Number of {e} for validation": val[val["label"] == args.classes[e]].shape[0]
    for e in list(args.classes.keys())
    }
    tags3 = {
    f"Number of {e} for testing": test[test["label"] == args.classes[e]].shape[0]
    for e in list(args.classes.keys())
    }
    tags = {**tags1, **tags2, **tags3}
    
    # Data augmentation
    train_tfms, test_tfms = get_augmentations(p=0.5)

    dataset_train = MyDataset(
        df_data=train, phase="train", data_dir=args.train_dir, transform=train_tfms
    )
    dataset_valid = MyDataset(
        df_data=val, phase="eval", data_dir=args.train_dir, transform=test_tfms
    )
    dataset_test = MyDataset(
        df_data=test, phase="test", data_dir=args.test_dir, transform=test_tfms
    )

    loader_train = DataLoader(
        dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    loader_valid = DataLoader(
        dataset=dataset_valid,
        batch_size=args.batch_size // 2,
        shuffle=True,
        num_workers=0,
    )
    loader_test = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size // 2,
        shuffle=True,
        num_workers=0,
    )
    return loader_train, loader_valid, loader_test, tags


def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def dict_to_sns(d):
    return SimpleNamespace(**d)


def set_tracking_uri():
    # Set tracking URI
    MODEL_REGISTRY = Path("experiments")
    Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    # Set experiment
    mlflow.set_experiment(experiment_name="baselines")


def running(args):
    # Tracking
    with mlflow.start_run(run_name="cnn") as run:
        # init data_loaders
        loader_train, loader_valid, loader_test, tags = init_data_loader(args)
        # load model
        model = SimpleCNN(args.num_classes).to(args.device)
        # Train & evaluate
        artifacts = train_cnn(args, model, loader_train, loader_valid, loader_test)

        # Log key metrics
        mlflow.log_metrics({"Accuracy": artifacts["performance"]["accuracy"]})
        mlflow.log_metrics({"precision": artifacts["performance"]["precision"]})
        mlflow.log_metrics({"recall": artifacts["performance"]["recall"]})
        mlflow.log_metrics({"f1": artifacts["performance"]["f1"]})
        mlflow.log_metrics({"best val loss": artifacts["best_val_loss"]})

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            save_dict(artifacts["performance"], Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)
        # set tags
        mlflow.set_tags(tags)
        # Log parameters
        mlflow.log_params(vars(artifacts["args"]))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/train_model.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Training classifcation model.")
    parser.add_argument(
        "--config-path",
        default="configs/train_model_config.json",
        help="path to config",
    )
    cmd_args = parser.parse_args()

    with open(cmd_args.config_path) as file:
        config = json.load(file)
        args = dict_to_sns(config)
    try:
        seed_everything(args.seed)
        set_tracking_uri()
        running(args)
    except Exception as e:
        print("Exception occured. Check logs.")
        logger.error(f"Failed to run script due to error:\n{e}")
        logger.error(traceback.format_exc())
