# Libraries
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn


def test(args, model, loader_test):
    model.eval()
    fn_list = []
    pred_list = []
    targets = []
    for x, fn, label in loader_test:
        with torch.no_grad():
            x = x.float()
            x = x.to(args.device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            fn_list += [n[:-4] for n in fn]
            pred_list += [p.item() for p in pred]
            targets += [p.item() for p in label]
    sub = pd.DataFrame({"id":fn_list, "pred":pred_list, "target": targets})
    # Evaluate (simple)
    metrics = precision_recall_fscore_support(
        sub["target"], sub["pred"], average="weighted"
    )
    acc = accuracy_score(sub["target"], sub["pred"])
    performance = {
        "accuracy": acc,
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
    }

    return performance