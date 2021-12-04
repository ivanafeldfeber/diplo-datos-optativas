#!/usr/bin/env bash

set -ex

if [ ! -d "./data/meli-challenge-2019/" ]
then
    mkdir -p ./data
    echo >&2 "Downloading Meli Challenge Dataset"
    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/meli-challenge-2019.tar.bz2 -o ./data/meli-challenge-2019.tar.bz2
    tar jxvf ./data/meli-challenge-2019.tar.bz2 -C ./data/
fi

if [ ! -f "./data/SBW-vectors-300-min5.txt.gz" ]
then
    mkdir -p ./data
    echo >&2 "Downloading SBWCE"
    curl -L https://cs.famaf.unc.edu.ar/\~ccardellino/resources/diplodatos/SBW-vectors-300-min5.txt.gz -o ./data/SBW-vectors-300-min5.txt.gz
fi

# Be sure the correct nvcc is in the path with the correct pytorch installation
export CUDA_HOME=/opt/cuda/10.1
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

python -m experiment.cnn1 \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.3 \
    --epochs 6 \
    --learning_rate 0.1

python -m experiment.cnn1 \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.3 \
    --learning_rate 0.01
    
python -m experiment.cnn1 \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --epochs 10 \
    --random_buffer_size 1024 \
    --batch_size 512

#CNN1
python -m experiment.cnn1 \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.5 \
    --learning_rate 0.0005 \
    --epochs 8 \
    --batch_size 1024

# Este experimento es con SGD, el resto con Adam como optimizador

python -m experiment.cnn2 \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.5 \
    --epochs 8 \
    --batch_size 512

python -m experiment.mlp_hyp \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.5
    --epochs 5

    
python -m experiment.cnn3 \
    --train-data ./data/meli-challenge-2019/spanish.train.jsonl.gz \
    --token-to-index ./data/meli-challenge-2019/spanish_token_to_index.json.gz \
    --pretrained-embeddings ./data/SBW-vectors-300-min5.txt.gz \
    --language spanish \
    --validation-data ./data/meli-challenge-2019/spanish.validation.jsonl.gz \
    --embeddings-size 300 \
    --hidden-layers 256 128 \
    --dropout 0.3 \
    --learning_rate 0.0005 \
    --epochs 10 \
    --batch_size 256
