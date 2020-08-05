import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from model.deepspeech import DeepSpeech
from utils.functions import data_processing, GreedyDecoder, IterMeter
from solver.solver import train, test
import numpy as np
import random
import argparse
import yaml

# Fix seed
seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Args
parser = argparse.ArgumentParser(description="Training script for DeepSpeech on Librispeech .")
parser.add_argument(
    "--config_path",
    metavar="config_path",
    type=str,
    help="Path to config file for training.",
    required=True,
)
parser.add_argument(
    "--experiment_name",
    metavar="experiment_name",
    type=str,
    help="Name for tensorboard experiment logs",
    default="",
)


def main(
    train_url="train-other-500", test_url="test-clean",
):

    writer = SummaryWriter(comment=args.experiment_name)
    # Load config file for experiment
    config_path = args.config_path
    print("---------------------------------------")
    print("Loading configure file at", config_path)
    with open(config_path, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    learning_rate = hparams["learning_rate"]
    batch_size = hparams["batch_size"]
    print(batch_size)
    epochs = hparams["epochs"]

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using GPU: {device}")
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    # Load datasets
    print("---------------------------------------")
    print("Loading datasets...", flush=True)
    train_dataset = data.ConcatDataset(
        [
            torchaudio.datasets.LIBRISPEECH("./data", url=path, download=True)
            for path in hparams["libri_train_set"]
        ]
    )
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        "./data", url=hparams["libri_test_set"], download=True
    )
    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=lambda x: data_processing(x, "train"),
        **kwargs,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        collate_fn=lambda x: data_processing(x, "valid"),
        **kwargs,
    )

    print("---------------------------------------")
    print("Creating model architecture...", flush=True)
    model = DeepSpeech(
        hparams["n_cnn_layers"],
        hparams["n_rnn_layers"],
        hparams["rnn_dim"],
        hparams["n_class"],
        hparams["n_feats"],
        hparams["stride"],
        hparams["dropout"],
    ).to(device)
    print(model)
    print("Num Model Parameters", sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams["learning_rate"])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["learning_rate"],
        steps_per_epoch=int(len(train_loader)),
        epochs=hparams["epochs"],
        anneal_strategy="linear",
    )
    scaler = GradScaler()
    if hparams["continue_from"]:
        print("Loading checkpoint model %s" % hparams["continue_from"])
        package = torch.load(hparams["continue_from"])
        model.load_state_dict(package["state_dict"])
        optimizer.load_state_dict(package["optim_dict"])
        start_epoch = int(package.get("epoch", 1))
    else:
        start_epoch = 1
    print("---------------------------------------")
    print("Training...", flush=True)
    iter_meter = IterMeter()
    for epoch in range(start_epoch, epochs + 1):
        training_loss = train(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            iter_meter,
            scaler,
            writer,
        )
        test_loss = test(model, device, test_loader, criterion, epoch, iter_meter, writer)
        if hparams["checkpoint"]:
            file_path = os.path.join("checkpoints", f"libri-epoch{epoch}.pth.tar")
            torch.save(
                model.serialize(
                    optimizer=optimizer, epoch=epoch, tr_loss=training_loss, val_loss=test_loss
                ),
                file_path,
            )
            print()
            print("Saving checkpoint model to %s" % file_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
