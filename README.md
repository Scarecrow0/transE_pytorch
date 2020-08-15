# TransE in PyTorch

## Introduction
Implementation of the TransE model for the knowledge graph representation.

## Code Structure

`utils.py` contains basic **Triple** class, its comparison, and other basic computations. It also contains the code for computing head-tail proportion for each relation, classifying relations into 1-1, 1-N, N-1 and N-N, and dividing triples according to them. For fairness, the evaluation is borrowed from previous work.


`evaluation.py` is evaluating the model in two metrics: meanrank and hits@10. We use multi-processing to speed up computation. For fairness, the evaluation is borrowed from previous work.

`data.py` contains various ways to generate negative triples and get a batch of training samples and its corresponding negative samples. For fairness, the evaluation is borrowed from previous work.


`models.py` contains the loss functions used in our algorithms, of which the most important is margin loss and orthogonal loss.

`train.py` The main logic(training/vaildation procedure) is written in this file. If you want to reproduce the result, you can execute this file. 


## Usage

Our programs are all written in Python 3.8, PyTorch 1.5. The GPU is required for training the model.

Usage:
python train.py -d DATASET_NAME

For the knowledges representation, we mainly use the WN11 and WN18 (WordNet) for expriments.

The model will be evaluated during training. For validation and LR scheduling, we will randomly take a batch of triples from the validation for evaluations, and use this validation loss for LR scheduling.
During the fix of epoch nums, will test model on test set. Evaluation result on the test set will be written into ./result/[dataset].txt, such as ./result/WN11.txt. 
