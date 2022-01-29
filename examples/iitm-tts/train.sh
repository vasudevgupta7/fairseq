#!/usr/bin/env bash

fairseq-train data-preprocessed \
--save-dir indic-tts-weights \
--config-yaml config.yaml \
--train-subset train \
--valid-subset train \
--n-frames-per-step 4 \
--task text_to_speech \
--arch tts_transformer \
--encoder-normalize-before \
--decoder-normalize-before \
--criterion tacotron2 \
--bce-pos-weight 5.0 \
--max-update 2000 \
--max-tokens 30000 \
--update-freq 2 \
--lr 2e-3 \
--clip-norm 5.0 \
--optimizer adam \
--lr-scheduler inverse_sqrt \
--warmup-updates 4000 \
--wandb-project iitm-tts \
--log-interval 16 \
--dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
--log-format tqdm
