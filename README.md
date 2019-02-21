# Machine Reading Comprehension
Machine Reading Comprehension in Tensorflow

## Prerequisites
* Python 3.6.6
* Tensorflow 1.12
* NumPy 1.15.4

## Usage
* Download [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) data
* Download [GloVe](https://nlp.stanford.edu/projects/glove/) or [FastText](https://research.fb.com/downloads/fasttext/) data
* Preprocess raw data
```bash
# preprocess train data
python squad/preprocess.py \
--format json \
--input_file data/squad/train-v1.1/train-v1.1.json \
--output_file data/squad/train-v1.1/train-v1.1.squad.json
# preprocess dev data
python squad/preprocess.py \
--format json \
--input_file data/squad/dev-v1.1/dev-v1.1.json \
--output_file data/squad/dev-v1.1/dev-v1.1.squad.json
```
* Run experiment
```bash
# run experiment in train_eval mode
# run experiment in train_eval mode
# run experiment in train_eval mode
```

## QANet
This repo contains a Tensorflow re-implementation of [QANet](https://arxiv.org/abs/1804.09541). QANet is a new Q&A architecture proposed by Google Brain, which does not require recurrent networks: Its encoder consists exclusively of convolution and self-attention, where convolution models local interactions and self-attention models global interactions.
