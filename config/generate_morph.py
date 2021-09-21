#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import logging
import os
import gc
import torch

from probing.config import Config


def generate_configs(config_fn):
    this_dir = os.path.dirname(__file__)
    data_root = os.path.join(this_dir, "..", "data")
    models = [
        "SZTAKI-HLT/hubert-base-cc",
        "tartuNLP/EstBERT",
        "TurkuNLP/bert-base-finnish-cased-v1",
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "bert-base-cased",
    ]
    for model in models:
        for lang_path in sorted(os.scandir(data_root), key=lambda p: p.name):
            language = lang_path.name
            for task_path in os.scandir(lang_path.path):
                task = task_path.name
                for subword in ['first', 'last']:
                    logging.info("=====================================")
                    logging.info(f"=== {model} {language} {task} {subword} ==")
                    logging.info("=====================================")
                    train_file = f"{task_path.path}/train.tsv"
                    dev_file = f"{task_path.path}/dev.tsv"
                    config = Config.from_yaml(config_fn)
                    config.model = 'SentenceRepresentationProber'
                    config.dataset_class = 'SentenceProberDataset'
                    config.subword_pooling = subword
                    config.layer_pooling = 'weighted_sum'
                    config.model_name = model
                    config.train_file = train_file
                    config.dev_file = dev_file
                    config.batch_size = 12
                    config.randomize_embedding_weights = True
                    config.train_base_model = True
                    config.remove_diacritics = False
                    yield config
                    gc.collect()
                    torch.cuda.empty_cache()
