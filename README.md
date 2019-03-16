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
* Search hyper-parameter
```bash
# random search hyper-parameters
python hparam_search.py --base-config config/config_mrc_template.xxx.json --search-config config/config_search_template.xxx.json --num-group 10 --random-seed 100 --output-dir config/search
```
* Visualize summary
```bash
# visualize summary via tensorboard
tensorboard --logdir=output
```
## Experiment
### QANet
[QANet](https://github.com/google-research/google-research/tree/master/qanet) is a MRC architecture proposed by Google Brain, which does not require recurrent networks: Its encoder consists exclusively of convolution and self-attention, where convolution models local interactions and self-attention models global interactions.

<p align="center"><img src="/reading_comprehension/document/QANet.architecture.png" width=700></p>
<p align="center"><i>Figure 1: An overview of the QANet architecture</i></p>

<p align="center"><img src="/reading_comprehension/document/QANet.metric.png" width=1000></p>
<p><i>Figure 2: The experiment details are reported on SQuAD v1 dataset. Both train & dev sets are processed using Spacy. Invalid samples are removed from both train & dev sets. EM results for QANet model with/without EMA are shown on left. F1 results for QANet model with/without EMA are shown on right</i></p>

|        Model        | # Epoch | # Train Steps | Batch Size |   Data Size   | # Head | # Dim |   EM   |   F1   |
|:-------------------:|:-------:|:-------------:|:----------:|:-------------:|:------:|:-----:|:------:|:------:|
| This implementation |    13   |    ~70,000    |     16     |  87k (no aug) |   8    |  128  |  70.2  |  80.0  |
|    Original Paper   |   ~13   |     35,000    |     32     |  87k (no aug) |   8    |  128  |   N/A  |  77.0  |
|    Original Paper   |   ~55   |    150,000    |     32     |  87k (no aug) |   8    |  128  |  73.6  |  82.7  |

<p><i>Table 1: The performance results are reported on SQuAD v1 dataset. Both train & dev sets are processed using Spacy. Invalid samples are removed from train set only. Settings for this QANet implementation is selected to be comparable with settings in original paper</i></p>

### BiDAF
[BiDAF](https://allenai.github.io/bi-att-flow/) (Bi-Directional Attention Flow) is a MRC architecture proposed by Allen Institute for Artificial Intelligence (AI2), which consists a multi-stage hierarchical process that represents the context at different levels of granularity and uses bidirectional attention flow mechanism to obtain a query-aware context representation without early summarization.

<p align="center"><img src="/reading_comprehension/document/BiDAF.architecture.png" width=700></p>
<p align="center"><i>Figure 3: An overview of the BiDAF architecture</i></p>

<p align="center"><img src="/reading_comprehension/document/BiDAF.metric.png" width=1000></p>
<p><i>Figure 4: The experiment details are reported on SQuAD v1 dataset. Both train & dev sets are processed using Spacy. Invalid samples are removed from both train & dev sets. EM results for BiDAF model with/without EMA are shown on left. F1 results for BiDAF model with/without EMA are shown on right</i></p>

|        Model        | # Epoch | # Train Steps | Batch Size | Attention Type | # Dim |   EM   |   F1   |
|:-------------------:|:-------:|:-------------:|:----------:|:--------------:|:-----:|:------:|:------:|
| This implementation |    12   |    ~17,500    |     60     |    trilinear   |  100  |  68.5  |  78.2  |
|    Original Paper   |    12   |    ~17,500    |     60     |    trilinear   |  100  |  67.7  |  77.3  |

<p><i>Table 2: The performance results are reported on SQuAD v1 dataset. Both train & dev sets are processed using Spacy. Invalid samples are removed from train set only. Settings for this BiDAF implementation is selected to be comparable with settings in original paper</i></p>

### R-Net
[R-Net](https://www.microsoft.com/en-us/research/publication/mcr/) is a MRC architecture proposed by Microsoft Research Asia (MSRA), which first matches the question and passage with gated attention-based recurrent networks to obtain the question-aware passage representation, then uses a self-matching attention mechanism to refine the representation by matching the passage against itself, and finally employs the pointer networks to locate the positions of answers from the passages.

<p align="center"><img src="/reading_comprehension/document/R-Net.architecture.png" width=700></p>
<p align="center"><i>Figure 5: An overview of the R-Net architecture</i></p>

## Reference
* Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V Le. [QANet: Combining local convolution with global self-attention for reading comprehension](https://arxiv.org/abs/1804.09541) [2018]
* Min Joon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. [Bidirectional attention flow for machine comprehension](https://arxiv.org/abs/1611.01603) [2017]
* Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang, and Ming Zhou. [Gated self-matching networks for reading comprehension and question answering](https://aclanthology.info/papers/P17-1018/p17-1018) [2017]
* Danqi Chen. [Neural reading comprehension and beyond](https://cs.stanford.edu/~danqi/papers/thesis.pdf) [2018]
