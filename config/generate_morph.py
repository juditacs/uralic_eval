#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import logging
from pathlib import Path
import gc
import torch

from probing.config import Config


def generate_configs(config_fn):
    data_root = Path(__file__).parent.parent / "data" / "morph"
    models = [
        "SZTAKI-HLT/hubert-base-cc",
        "tartuNLP/EstBERT",
        "TurkuNLP/bert-base-finnish-cased-v1",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "bert-base-cased",
    ]
    for model in models:
        for lang_path in data_root.iterdir():
            language = lang_path.name
            for task_path in lang_path.iterdir():
                task = task_path.name
                for subword in ['first', 'last']:
                    logging.info("=====================================")
                    logging.info(f"=== {model} {language} {task} {subword} ==")
                    logging.info("=====================================")
                    train_file = str(task_path / "train.tsv")
                    dev_file = str(task_path / "dev.tsv")
                    config = Config.from_yaml(config_fn)
                    config.model = 'SentenceRepresentationProber'
                    config.dataset_class = 'SentenceProberDataset'
                    config.subword_pooling = subword
                    config.layer_pooling = 'weighted_sum'
                    config.model_name = model
                    config.train_file = train_file
                    config.dev_file = dev_file
                    config.batch_size = 12
                    config.randomize_embedding_weights = False
                    config.train_base_model = False
                    config.remove_diacritics = False
                    yield config
                    gc.collect()
                    torch.cuda.empty_cache()
