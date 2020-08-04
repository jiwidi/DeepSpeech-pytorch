import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter,writer):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        # experiment.log_metric("loss", loss.item(), step=iter_meter.get())
        # experiment.log_metric("learning_rate", scheduler.get_lr(), step=iter_meter.get())
        writer.add_scalar("Loss/train", loss.item(), iter_meter.get())
        writer.add_scalar("learning_rate",scheduler.get_last_lr()[0],iter_meter.get())
        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(spectrograms),
                    data_len,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader, criterion, epoch, iter_meter,writer):
    print("\nevaluating...")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(
                output.transpose(0, 1), labels, label_lengths
            )
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    # experiment.log_metric("test_loss", test_loss, step=iter_meter.get())
    # experiment.log_metric("cer", avg_cer, step=iter_meter.get())
    # experiment.log_metric("wer", avg_wer, step=iter_meter.get())
    writer.add_scalar("test_loss",test_loss,iter_meter.get())
    writer.add_scalar("cer",avg_cer,iter_meter.get())
    writer.add_scalar("wer",avg_wer,iter_meter.get())
    print(
        "Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            test_loss, avg_cer, avg_wer
        )
    )

