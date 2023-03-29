Code and result tables for the ["Evaluating Transferability of BERT Models on Uralic Languages"](https://arxiv.org/abs/2109.06327) IWCLUL2021 paper

# Prerequisites

The source code for running the experiments is available in the [probing](https://github.com/juditacs/probing) package.
This repository only contains the data, the configuration files, the results tables and the analysis notebook.

# Training a single experiment

Set `PROBING_PATH` to wherever you downloaded [probing](https://github.com/juditacs/probing).

## Morphology

Finetuning:

    python $PROBING_PATH/src/probing/train.py \
        --config config/morphology_finetuning.yaml \
        --train data/morph/Hungarian/person_verb/train.tsv \
        --dev data/morph/Hungarian/person_verb/dev.tsv

No finetuning:

    python $PROBING_PATH/src/probing/train.py \
        --config config/morphology.yaml \
        --train data/morph/Hungarian/person_verb/train.tsv \
        --dev data/morph/Hungarian/person_verb/dev.tsv

## POS and NER

Finetuning:

    python $PROBING_PATH/src/probing/train.py \
        --config config/tagging_finetuning.yaml \
        --train data/pos/Estonian/train \
        --dev data/pos/Estonian/dev

No finetuning:

    python $PROBING_PATH/src/probing/train.py \
        --config config/tagging.yaml \
        --train data/pos/Estonian/train \
        --dev data/pos/Estonian/dev

# Train multiple experiments

Morphology experiments without finetuning can be run consecutively with a cached model that allows for a significant speed up.

    python $PROBING_PATH/src/probing/train_many_configs.py \
        -p config/generate_morph.py \
        -c config/common.yaml
