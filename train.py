"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from project.model.deepspeech_main import DeepSpeech

seed_everything(42)


def main(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = DeepSpeech(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    # Each LightningModule defines arguments relevant to it
    parser = DeepSpeech.add_model_specific_args(parent_parser, root_dir)
    # Data
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--data_root", default="data/", type=str)
    parser.add_argument("--data_train", default=["train-clean-100", "train-clean-360", "train-other-500"])
    parser.add_argument("--data_test", default=["test-clean"])

    # training params (opt)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    #parser.add_argument("--accumulate_grad_batches", default=40, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--precission", default=32, type=int)
    parser.add_argument("--gradient_clip", default=0, type=float)
    parser.add_argument("--auto_scale_batch_size", default=False, type=bool)
    parser.add_argument("--auto_select_gpus", default=True, type=bool)
    parser.add_argument("--log_gpu_memory", default=True, type=bool)
    parser.add_argument("--use_amp", default=False, type=bool) #update when getting ampere
    parser.add_argument("--early_stop_metric", default="wer", type=str)
    parser.add_argument("--early_stop_patience", default=3, type=int)
    parser.add_argument("--experiment_name", default="DeepSpeech", type=str)
    parser.add_argument("--loggs_path", default="runs", type=str)
    # callbacks

    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Early stopper
    early_stop = EarlyStopping(
        monitor=args.early_stop_metric,
        patience=args.early_stop_patience,
        verbose=True,
    )
    #Logger
    logger = TensorBoardLogger(save_dir=args.loggs_path, name=args.experiment_name)

    args.early_stop_callback = early_stop
    args.logger = logger
    #setattr(args, "accumulate_grad_batches", 40)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == "__main__":
    run_cli()
