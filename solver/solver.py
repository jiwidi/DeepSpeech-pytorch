import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from utils.functions import GreedyDecoder, cer, wer
import time


def train(
    model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, scaler, writer
):
    """Performs a full epoch training loop on a given model

    Args:
        model (DeepSpeech): Model being trainged
        device (Torch device): Device where the train happens
        train_loader (Dataloader): Dataloader for the training data
        criterion (Loss function): Loss function for the training
        optimizer (Optimizer): Optimizer object
        scheduler (Learning rate scheduler): Scheduler for the learning rate
        epoch (Int): Epoch number for log training
        iter_meter (Iter_meter): Iter meter object to keep track of each iteration
        scaler (Scaler): Pytorch autograd scaler for mixed precission training
        writer (Tensorboard writer): Tensorboard writer to write logs to

    Returns:
        Float: Loss of the last minibatch
    """
    model.train()
    data_len = len(train_loader.dataset)
    train_start_time = time.time()
    for batch_idx, _data in enumerate(train_loader):

        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)

        # Mixed precision
        scaler.scale(loss).backward()  # loss.backward()

        scaler.step(optimizer)  # optimizer.step()
        scheduler.step()
        iter_meter.step()

        # Updates the scale for next iteration
        scaler.update()

        writer.add_scalar("Loss/train", loss.item(), iter_meter.get())
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], iter_meter.get())

        if batch_idx % 100 == 0 or batch_idx == data_len:
            train_finish_time = time.time()
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTook: {:.2f} seconds".format(
                    epoch,
                    batch_idx * len(spectrograms),
                    data_len,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    train_finish_time - train_start_time,
                )
            )
            train_start_time = time.time()
    return loss.item()


def test(model, device, test_loader, criterion, epoch, iter_meter, writer):
    """Function to calculate test metrics (Character error rate, Word error rate)
    given a dataloader

    Args:
        model (DeepSpeech): The model used to calculate test metrics
        device (Torch device): Device to run the inference on
        test_loader (Dataloader): Pytorch dataloader
        criterion (Loss function): Loss function
        epoch (Int): Epoch number for tensoboard logs
        iter_meter (Iter_meter): Object to keep track
        writer (Tensoboard writer): Tensorboard object to write logs to

    Returns:
        Float: Test data loss
    """
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

    writer.add_scalar("Loss/test", test_loss, iter_meter.get())
    writer.add_scalar("Test_metrics/cer", avg_cer, epoch)
    writer.add_scalar("Test_metrics/wer", avg_wer, epoch)
    print(
        "Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            test_loss, avg_cer, avg_wer
        )
    )
    return test_loss
