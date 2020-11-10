"""
Example template for defining a system.
"""
from argparse import ArgumentParser

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchaudio
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader

from project.utils.functions import data_processing, GreedyDecoder, cer, wer
from project.utils.cosine_annearing_with_warmup import CosineAnnealingWarmUpRestarts


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
    except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class DeepSpeech(LightningModule):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, **kwargs):
        super(DeepSpeech, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = self.hparams.learning_rate
        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(
            *[ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) for _ in range(n_cnn_layers)]
        )
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        self.birnn_layers = nn.Sequential(
            *[
                BidirectionalGRU(
                    rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                    hidden_size=rnn_dim,
                    dropout=dropout,
                    batch_first=i == 0,
                )
                for i in range(n_rnn_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
        )

        self.criterion = nn.CTCLoss(blank=28)
        self.example_input_array = torch.rand(8, 1, 128, 1151)

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

    def serialize(self, optimizer, epoch, tr_loss, val_loss):
        package = {
            "state_dict": self.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        if tr_loss is not None:
            package["tr_loss"] = tr_loss
            package["val_loss"] = val_loss
        return package

    # ---------------------
    # Pytorch lightning overrides
    # ---------------------
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        spectrograms, labels, input_lengths, label_lengths = batch
        y_hat = self(spectrograms)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        tensorboard_logs = {"Loss/train": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        spectrograms, labels, input_lengths, label_lengths = batch
        y_hat = self(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = self.criterion(output, labels, input_lengths, label_lengths)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        n_correct_pred = sum([int(a == b) for a, b in zip(decoded_preds, decoded_targets)])

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor([sum(test_wer) / len(test_wer)])  # Need workt to make all operations in torch
        logs = {
            "cer": avg_cer,
            "wer": avg_wer,
        }
        return {
            "val_loss": loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(spectrograms),
            "log": logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def test_step(self, batch, batch_idx):
        spectrograms, labels, input_lengths, label_lengths = batch
        y_hat = self(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(y_hat, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = self.criterion(output, labels, input_lengths, label_lengths)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
        n_correct_pred = sum([int(a == b) for a, b in zip(decoded_preds, decoded_targets)])

        test_cer, test_wer = [], []
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

        avg_cer = torch.FloatTensor([sum(test_cer) / len(test_cer)])
        avg_wer = torch.FloatTensor([sum(test_wer) / len(test_wer)])  # Need workt to make all operations in torch
        logs = {
            "Metrics/cer": avg_cer,
            "Metrics/wer": avg_wer,
        }
        return {
            "val_loss": loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(spectrograms),
            "log": logs,
            "wer": avg_wer,
            "cer": avg_cer,
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(x["n_pred"] for x in outputs)
        avg_wer = torch.stack([x["wer"] for x in outputs]).mean()
        avg_cer = torch.stack([x["cer"] for x in outputs]).mean()
        tensorboard_logs = {"Loss/val": avg_loss, "val_acc": val_acc, "Metrics/wer": avg_wer, "Metrics/cer": avg_cer}
        return {"val_loss": avg_loss, "log": tensorboard_logs, "wer": avg_wer, "cer": avg_cer}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(x["n_pred"] for x in outputs)
        avg_wer = torch.stack([x["wer"] for x in outputs]).mean()
        avg_cer = torch.stack([x["cer"] for x in outputs]).mean()
        tensorboard_logs = {"Loss/test": avg_loss, "test_acc": test_acc, "Metrics/wer": avg_wer, "Metrics/cer": avg_cer}
        return {"test_loss": avg_loss, "log": tensorboard_logs, "wer": avg_wer, "cer": avg_cer}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate/10)
        # lr_scheduler = {'scheduler':optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.hparams.learning_rate/5,max_lr=self.hparams.learning_rate,step_size_up=2000,cycle_momentum=False),
        # lr_scheduler = {# 'scheduler': optim.lr_scheduler.OneCycleLR(
        #                             #     optimizer,
        #                             #     max_lr=self.learning_rate,
        #                             #     steps_per_epoch=int(len(self.train_dataloader())),
        #                             #     epochs=self.hparams.epochs,
        #                             #     anneal_strategy="linear",
        #                             #     final_div_factor = 0.06,
        #                             #     pct_start = 0.04
        #                             # ),
        #                 'scheduler': CosineAnnealingWarmUpRestarts(optimizer, T_0=int(len(self.train_dataloader())*math.pi), T_mult=2, eta_max=self.learning_rate,  T_up=int(len(self.train_dataloader()))*2, gamma=0.8),
        #                 'name': 'learning_rate', #Name for tensorboard logs
        #                 'interval':'step',
        #                 'frequency': 1}

        return [optimizer] #, [lr_scheduler]

    def prepare_data(self):
        if not os.path.exists(self.hparams.data_root):
            os.makedirs(self.hparams.data_root)
        a = [
            torchaudio.datasets.LIBRISPEECH(self.hparams.data_root, url=path, download=True)
            for path in self.hparams.data_train
        ]
        b = [
            torchaudio.datasets.LIBRISPEECH(self.hparams.data_root, url=path, download=True)
            for path in self.hparams.data_test
        ]
        return a,b

    def setup(self, stage):
        self.train_data = data.ConcatDataset(
            [
                torchaudio.datasets.LIBRISPEECH(self.hparams.data_root, url=path, download=True)
                for path in self.hparams.data_train
            ]
        )
        self.test_data = data.ConcatDataset(
            [
                torchaudio.datasets.LIBRISPEECH(self.hparams.data_root, url=path, download=True)
                for path in self.hparams.data_test
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda x: data_processing(x, "train"),
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, "valid"),
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda x: data_processing(x, "valid"),
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument("--n_cnn_layers", default=3, type=int)
        parser.add_argument("--n_rnn_layers", default=5, type=int)
        parser.add_argument("--rnn_dim", default=512, type=int)
        parser.add_argument("--n_class", default=29, type=int)
        parser.add_argument("--n_feats", default=128, type=str)
        parser.add_argument("--stride", default=2, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)

        return parser
