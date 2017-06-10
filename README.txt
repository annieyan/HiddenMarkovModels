************************************************
Hidden Markov Models for POS tagging
Data: twitter data with POS tags
Smoothing method: linear interpolation
Decoding: Viterbi 

author : An Yan
date: Feb, 2017
Python version: 2.7
*************************************************

HOW TO RUN:

example:

python yanan_hmm.py -t 5 twt.train.json twt.dev.json

** this will output accuracy/confusion matrix of both
** bigram and trigram hmm on dev data with pre-trained
** best lamda.lamda =(0.001, 0.999) for bigram_hmm and
** lamda = (0.6, 0.3, 0.1) for trigram hmm

****************************
parameters: 

-t: unk threshold, default = 1. 
test data set 
dev data set or test data set.

*****************************
Help message, please type


python yanan_lm.py -h

************************************************
note: 
if you run "python yanan_hmm.py -t 5 twt.train.json twt.dev.json",
you will get accuracies of bigram and trigram hmm on dev data only.
****************************************************************
