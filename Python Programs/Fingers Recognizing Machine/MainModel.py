import torch.nn as nn
import torch
import lightning as L
from torchmetrics import Accuracy

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module(
            name='conv1',
            module=nn.Conv2d(
                in_channels=1, out_channels=32,
                kernel_size=5, padding=2
            )
        )
        self.model.add_module(
            name='relu1',
            module=nn.ReLU()
        )
        self.model.add_module(
            name='pool1',
            module=nn.MaxPool2d(kernel_size=2)
        )
        self.model.add_module(
            name='conv2',
            module=nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=5, padding=2
            )
        )
        self.model.add_module(
            name='relu2',
            module=nn.ReLU()
        )
        self.model.add_module(
            name='pool2',
            module=nn.MaxPool2d(kernel_size=2)
        )
        self.model.add_module(
            name='flatten',
            module=nn.Flatten()
        )
        self.model.add_module(
            name='fc1',
            module=nn.Linear(4 * 128 ** 2, 1024))
        self.model.add_module(
            name='relu3',
            module=nn.ReLU()
        )
        self.model.add_module(
            name='dropout',
            module=nn.Dropout(p=0.5)
        )
        self.model.add_module(
            name='fc2',
            module=nn.Linear(1024, 6)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
    
class LightningNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = NN()

        self.train_acc = Accuracy(task="multiclass", num_classes=6)
        self.valid_acc = Accuracy(task="multiclass", num_classes=6)
        self.test_acc = Accuracy(task="multiclass", num_classes=6)

        
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = nn.functional.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        self.train_acc.update(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = nn.functional.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        self.valid_acc.update(pred, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = nn.functional.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        self.test_acc.update(pred, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer