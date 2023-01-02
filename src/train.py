import os
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments
from transformers.integrations import ClearMLCallback

from callbacks import CustomClearMLCallback
from dataset import create_dataset, hashed_train_test_split, load_csv
from metrics import compute_metrics
from model import load_model
from utils import configure_clearml, instantiate_entities


def get_experiment_name(base_name: str):
    current_time = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    experiment_name = base_name + "-" + current_time
    return experiment_name


def train(config: DictConfig):
    # Load model
    model, tokenizer = load_model(config.model_name)

    # Load dataset
    csv_train_path = os.path.join(config.dataset_path, "train.csv")
    df_train = load_csv(csv_train_path, preprocess=True, use_cache=config.use_cache_csv)

    # Split & create datasets
    df_train, df_valid = hashed_train_test_split(df_train, config.validation_size)
    ds_train = create_dataset(df_train, tokenizer)
    ds_valid = create_dataset(df_valid, tokenizer)

    # Load callbacks
    callbacks = instantiate_entities(config.callbacks)

    # Create a trainer object
    training_args = TrainingArguments(output_dir=config.experiment_path, **config.training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # # Pop default ClearML callback.
    # # It reports models automatically, which is b-a-d
    if config.training_args.report_to != "none":
        trainer.pop_callback(ClearMLCallback)
        trainer.add_callback(CustomClearMLCallback)

    # Start training
    trainer.train()

    # Save best model at the end
    trainer.save_model("best_model")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig):
    # Get paths
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    project_name = config.project_name
    clearml_file = config.clearml_file

    experiment_name = get_experiment_name(config.experiment_name)
    experiment_path = os.path.join(config.output_path, experiment_name)
    config.experiment_path = experiment_path

    # Configure ClearML
    if config.training_args.report_to != "none":
        configure_clearml(project_name, experiment_name, clearml_file)

    # Train the model
    train(config)


if __name__ == "__main__":
    main()
