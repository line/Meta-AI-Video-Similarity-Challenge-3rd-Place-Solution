import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LitModel(pl.LightningModule):
    def __init__(self, model_name, labels):
        super().__init__()

        self.labels = labels
        num_classes = len(labels)

        if model_name.startswith("hf-hub:"):
            model = timm.create_model(model_name, pretrained=True)
            model.head[-1] = nn.Linear(1024, num_classes)
            self.model = model
        else:
            self.model = timm.create_model(
                model_name, num_classes=num_classes, pretrained=True
            )

        self.criterion = nn.BCEWithLogitsLoss()

        self.valid_acc = torchmetrics.MultioutputWrapper(
            torchmetrics.Accuracy("binary", average=None),
            num_classes,
        )
        self.valid_auc = torchmetrics.MultioutputWrapper(
            torchmetrics.AUROC("binary", average=None),
            num_classes,
        )
        self.valid_ap = torchmetrics.MultioutputWrapper(
            torchmetrics.AveragePrecision("binary", average=None),
            num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.criterion(logit, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        y_hat = torch.sigmoid(logit)

        loss = self.criterion(logit, y)

        self.valid_acc.update(y_hat, y)
        self.valid_auc.update(y_hat, y)
        self.valid_ap.update(y_hat, y.long())

        self.log("valid_loss", loss)

    def validation_epoch_end(self, outputs):
        acc = self.valid_acc.compute()
        auc = self.valid_auc.compute()
        ap = self.valid_ap.compute()

        self.log_dict(
            {f"valid_{l}_acc": x for l, x in zip(self.labels, acc)}, sync_dist=True
        )
        self.log_dict(
            {f"valid_{l}_auc": x for l, x in zip(self.labels, auc)}, sync_dist=True
        )
        self.log_dict(
            {f"valid_{l}_ap": x for l, x in zip(self.labels, ap)}, sync_dist=True
        )

        self.valid_acc.reset()
        self.valid_auc.reset()
        self.valid_ap.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
