import numpy as np
import mlflow
import copy
import torch
import torch.nn as nn
from test import test

# Train the model
def train_cnn(args, model, loader_train, loader_valid, loader_test):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
    len(loader_train)
    min_valid_loss = np.inf
    for epoch in range(args.num_epochs):
        train_loss = 0.0
        for i, (images, labels) in enumerate(loader_train):
            images = images.to(args.device)
            images = images.float()
            labels = labels.to(args.device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        model.eval()
        for i, (images, labels) in enumerate(loader_valid):
            images = images.to(args.device)
            images = images.float()
            labels = labels.to(args.device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
        # Logging
        print(
            f"Epoch {epoch} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}"
        )
        # Tracking
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": valid_loss}, step=epoch
        )
        if min_valid_loss > valid_loss:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
            )
            min_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
    performance = test(args, best_model, loader_test)
    return {
        "args": args,
        "model": best_model,
        "performance": performance,
        "best_val_loss": min_valid_loss,
    }
