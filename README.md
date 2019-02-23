# Machine Reading Comprehension
Machine reading comprehension (MRC), a task which asks machine to read a given context then answer questions based on its understanding, is considered one of the key problems in artificial intelligence and has significant interest from both academic and industry. Over the past few years, great progress has been made in this field, thanks to various end-to-end trained neural models and high quality datasets with large amount of examples proposed. In this repo, I'll share more details on MRC task by re-implementing a few MRC models and testing them on standard MRC datasets.

## Setting
* Python 3.6.6
* Tensorflow 1.12
* NumPy 1.15.4
* NLTK 3.3
* Spacy 2.0.12

## DataSet
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowd-workers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
* [GloVe](https://nlp.stanford.edu/projects/glove/) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

## Usage
* Preprocess data
```bash
# preprocess train data
python squad/preprocess.py --format json --input_file data/squad/train-v1.1/train-v1.1.json --output_file data/squad/train-v1.1/train-v1.1.squad.json
# preprocess dev data
python squad/preprocess.py --format json --input_file data/squad/dev-v1.1/dev-v1.1.json --output_file data/squad/dev-v1.1/dev-v1.1.squad.json
```
* Run experiment
```bash
# run experiment in train + eval mode
python reading_comprehension_run.py --mode train_eval --config config/config_mrc_template.xxx.json
# run experiment in train only mode
python reading_comprehension_run.py --mode train --config config/config_mrc_template.xxx.json
# run experiment in eval only mode
python reading_comprehension_run.py --mode eval --config config/config_mrc_template.xxx.json
```
* Visualize summary
```bash
# visualize summary via tensorboard
tensorboard --logdir=output
```
## Experiment
### QANet
[QANet](https://arxiv.org/abs/1804.09541) is a MRC architecture proposed by Google Brain, which does not require recurrent networks: Its encoder consists exclusively of convolution and self-attention, where convolution models local interactions and self-attention models global interactions.

<img src="/reading_comprehension/document/QANet.architecture.png" width=700><br />
*Figure 1: An overview of the QANet architecture*

<img src="/reading_comprehension/document/QANet.metric.png" width=1000><br />
*Figure 2: The performance results reported are on SQuAD v1 dataset. Both train & dev sets are processed using Spacy and invalid samples are removed. EM results for QANet model with/without EMA are shown on left. F1 results for QANet model with/without EMA are shown on right*

### BiDAF
[BiDAF](https://allenai.github.io/bi-att-flow/) (Bi-Directional Attention Flow) is a MRC architecture proposed by Allen Institute for Artificial Intelligence (AI2), which consists a multi-stage hierarchical process that represents the context at different levels of granularity and uses bidirectional attention flow mechanism to obtain a query-aware context representation without early summarization.

<img src="/reading_comprehension/document/BiDAF.architecture.png" width=700><br />
*Figure 3: An overview of the BiDAF architecture*

<img src="/reading_comprehension/document/BiDAF.metric.png" width=1000><br />
*Figure 4: The performance results reported are on SQuAD v1 dataset. Both train & dev sets are processed using Spacy and invalid samples are removed. EM results for BiDAF model with/without EMA are shown on left. F1 results for BiDAF model with/without EMA are shown on right*

### R-Net
[R-Net](https://www.microsoft.com/en-us/research/publication/mcr/) is a MRC architecture proposed by Microsoft Research Asia (MSRA), which first matches the question and passage with gated attention-based recurrent networks to obtain the question-aware passage representation, then uses a self-matching attention mechanism to refine the representation by matching the passage against itself, and finally employs the pointer networks to locate the positions of answers from the passages.

<img src="/reading_comprehension/document/R-Net.architecture.png" width=700><br />
*Figure 5: An overview of the R-Net architecture*
