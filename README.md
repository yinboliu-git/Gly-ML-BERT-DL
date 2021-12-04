# Gly-ML-BERT-DL
### Please download the BERT model in advance
BERT-Base: [https://github.com/google-research/bert](https://github.com/google-research/bert)   Based on BERT_ Base's 12 layer natural language pre training model

BERT-PROT: [https://github.com/JianyuanLin/Bert-Protein/](https://github.com/JianyuanLin/Bert-Protein/)  Protein pre training model, based on the pre training model with word segmentation of 1

BERT-Tape: [https://github.com/songlab-cal/tape/](https://github.com/songlab-cal/tape/)  Please use visionvision bert-base(Transformer model)

### First, use "Bert_extraction" for feature extraction

Use the command in a to extract basic data from the dataset.

Different methods are used to extract features from basic data

### Second, use the deep learning model (Gly_DL) to read the feature data and train

use_ANN_gridE.py & use_ANN_gridE_tape.py is the code of the training model

use_ANN_indp_gridE.py is the code tested on independent test data using the model

### Third, use machine learning code to train the model

First use. / Gly_ ML/feature_ Extract.py extract features

and use. / Gly_ ML/grid_ fold_ singel_ feature_ Model.py training model.

### Finally, use plt_img code to draw

plt_roc_train.py:  roc img

plt_zhifang.py: histogram img

t-SNE.py: t-SNE img
