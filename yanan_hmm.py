# ---- HMM for POS tagging ------#
# ---- input twitter data with POS taggs -------#
from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import sklearn 
import string
from collections import deque
from itertools import islice
import collections
import math
import argparse
import time
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import matplotlib.pyplot as plt
import itertools

# GLOABLS
STOP_token = '_STOP_'
UNK_token = '_UNK_' 
START_token = '_START_'
ADD_K_SMOOTHING = 'add_k_smoothing'
LINER_INT = 'liner interpolation'
NO_SMOOTHING = 'no smoothing'

# special tweet char
MENTION_token = '_AT_'
HASHTAG_token = '_HASHTAG_'
RT_token = '_RT_'
EMOJIS_token = '_EMOJIS_'
URL_token = '_URL_'
SMILEYS_token = '_SMILEYS_'
NUMBERS_token = '_NUMBERS_'



class patterns:
    URL_PATTERN=re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    HASHTAG_PATTERN = re.compile(r'#\w*')
    MENTION_PATTERN = re.compile(r'@\w*')
    RT_PATTERN = re.compile(r'^(RT|FAV)')
    #EMOJIS_PATTERN = re.compile(b'([\\xww\\w*])')
    try:
            # UCS-4
        EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
            # UCS-2
        EMOJIS_PATTERN = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')

    SMILEYS_PATTERN = re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE)
    NUMBERS_PATTERN = re.compile(r"(^|\s)(\-?\d+(?:\.\d)*|\d+)")




class HMM:
    def __init__(self,args):
        #  for tags
        self.vocabulary_tag = set()
        self.tag_space = list()       
        self.total_words_len_tag = 0

        # tag unigram count
        self.unigram_count = None
        #self.unigram_V = None
        self.unigrams_prob_dict = None
        self.V = 0
        self.trigram_prob_dict = None
        self.bigram_prob_dict = None

        # for texts
        self.replaced_tokens_train = list()
        self.vocabulary_text = set()

        # these are source file paths, json or text
        self.training_set = args.training_set
        self.dev_set = args.dev_set

        self.unk_threshold = args.threshold

        # following for processed json
        self.train_tweets = list()
        # for count [tweet: tag] pair
        self.flat_train_tweets = list()
        self.test_tweets = list()
        self.train_tag_tokens = list()
        # {(tweet,tag):count}
        self.tag_pair_count = dict()
        # replaced UNK tag pair count
        self.replaced_pair_count = dict()
      
        self.train_text_tokens = list()
        self.test_tag_tokens = list()
        self.test_text_tokens = list()
        self.emit_prob = dict()
        #self.replaced_train_tags = list()

        
    '''
    open a json file
    the input is a list of list
    return a list of list
    '''
    def json_read(self,filename):
        tweets = list()
        with open(filename,'r+') as f:
            for line in f:
                tmp_tp = tuple()
                tmp_tp+=tuple(json.loads(line.decode("UTF-8")))  
                tweets.append(tmp_tp) 
        return tweets

    '''
    take tweets as list of list, add STOP to texts and tages
    return a tuple(text tokens, tag tokens)
    '''
    def json_process(self,tweets):
        text_token = list()
        tag_token = list()
        
        for sent in tweets:
            for word in sent:
             # add stop 
                text_token.append(word[0])
                tag_token.append(word[1])
                self.flat_train_tweets.append(tuple(word))
            #text_token.append(STOP_token)
            tag_token.append(STOP_token)
        self.tag_pair_count = Counter(self.flat_train_tweets)
        print("text, tag seperation done")
        return text_token,tag_token



    '''
    open a text file, add STOP_token, replace punctuations
    '''
    def text_processing(self,filename):
        text = ""
        with open(filename,'r+') as f:
            text = f.read()
            #print text
            text = text.replace('\n',' '+STOP_token+'\n')
            for ch in '!"#$%&()*+,-./:;<=>?@[\\]^`{|}~':
                text = string.replace(text, ch, ' ')       
        return text


    '''tokenize processed text'''
    def tokenize_unigram(self,text):
        return nltk.word_tokenize(text)


    '''get Vocabulary from training text set. including UNK
    :param: input tokenized training text, type: list 
    :output: unigrams {unigrams:count}
    This in hmm only for replacing UNK in [tweet:tag] pairs
    Still need to deal with special characters and recognized entity
    #,@,emoji,Titlecase, RT
    '''
    def unigram_V(self,train_token):
            # this is the total length/num of tokens in training data
            #global total_words_len
            #global replaced_tokens_train         
            total_text_token_len = len(train_token)
            # initialize word_count pairs
            unigram_V = {}
            unigram_V[UNK_token] = 0
             # need to deal with emoji/#/@ and named entity here-----
            # initial work-count dict population
            for token in train_token:
                # check numbers, mention, RT.....
                if bool(patterns.NUMBERS_PATTERN.match(token)):
                    unigram_V[NUMBERS_token]= unigram_V.get(NUMBERS_token,0) + 1
                elif bool(patterns.SMILEYS_PATTERN.match(token)):
                    unigram_V[SMILEYS_token]= unigram_V.get(SMILEYS_token,0) + 1
                elif bool(patterns.URL_PATTERN.match(token)):
                    unigram_V[URL_token]= unigram_V.get(URL_token,0) + 1
                elif bool(patterns.EMOJIS_PATTERN.match(token)):
                    unigram_V[EMOJIS_token]= unigram_V.get(EMOJIS_token,0) + 1
                elif bool(patterns.RT_PATTERN.match(token)):
                    unigram_V[RT_token]= unigram_V.get(RT_token,0) + 1
                elif bool(patterns.HASHTAG_PATTERN.match(token)):
                    unigram_V[HASHTAG_token]= unigram_V.get(HASHTAG_token,0) + 1
                elif bool(patterns.MENTION_PATTERN.match(token)):
                    unigram_V[MENTION_token]= unigram_V.get(MENTION_token,0) + 1
                else:
                    unigram_V[token]= unigram_V.get(token,0) + 1         
        
            # re-assign UNK
            unk_words = set()
            items = unigram_V.iteritems()
            for word, count in items:
                # treat low freq word as UNK
                if count <= self.unk_threshold:
                    unk_words.add(word)
                    unigram_V[UNK_token] += count
            
            unk_words.discard(STOP_token)
            unk_words.discard(UNK_token)

            for word in unk_words:
                del unigram_V[word]

            self.replaced_tokens_train = list(unigram_V.keys())
            for idx, token in enumerate(self.replaced_tokens_train):            
                if token in unk_words:                
                    self.replaced_tokens_train[idx] = UNK_token
      
            # refresh tag_pair count
            self.replaced_pair_count = self.tag_pair_count.copy()
            pair_item = self.tag_pair_count.iteritems()
            for pair,count in pair_item:
                # e.g., {[@taran,@]:4}  ->{[_AT_,@]:n+4}  
                #print ('pair[0]',pair[0], pair[1])
                if bool(patterns.NUMBERS_PATTERN.match(pair[0])):
                    tmp_tp = NUMBERS_token,pair[1]
                    self.replaced_pair_count[tmp_tp] += count
                    self.replaced_pair_count.pop(pair)
                    #print(self.replaced_pair_count[tmp_tp], count)

                elif bool(patterns.SMILEYS_PATTERN.match(pair[0])):
                    tmp_tp = SMILEYS_token,pair[1]
                    self.replaced_pair_count[tmp_tp] += count
                    self.replaced_pair_count.pop(pair)

                elif bool(patterns.URL_PATTERN.match(pair[0])):
                    tmp_tp = URL_token,pair[1]
                    self.replaced_pair_count[tmp_tp] += count
                    self.replaced_pair_count.pop(pair)

                elif bool(patterns.EMOJIS_PATTERN.match(pair[0])):
                    tmp_tp = EMOJIS_token,pair[1]
                    self.replaced_pair_count[tmp_tp] += count        
                    self.replaced_pair_count.pop(pair)
                elif bool(patterns.RT_PATTERN.match(pair[0])):
                    tmp_tp = RT_token,pair[1]
                    self.replaced_pair_count[tmp_tp] += count          
                    self.replaced_pair_count.pop(pair)

                elif bool(patterns.HASHTAG_PATTERN.match(pair[0])):
                    tmp_tp = HASHTAG_token,pair[1]
                    self.replaced_pair_count[tmp_tp] += count   
                    self.replaced_pair_count.pop(pair)

                elif bool(patterns.MENTION_PATTERN.match(pair[0])):
                    tmp_tp = MENTION_token,pair[1]
                    self.replaced_pair_count[tmp_tp] += count   
                    self.replaced_pair_count.pop(pair)
                    #print (tmp_tp)

                elif pair[0] in unk_words:
                    # update Key  
                    tmp_tp = UNK_token,pair[1]
                    self.replaced_pair_count[tmp_tp]+= count

                    self.replaced_pair_count.pop(pair)  


                    #################################
            print (unigram_V[MENTION_token],unigram_V[URL_token])
            print('word count pair',self.replaced_pair_count[MENTION_token,'@'])
            return unigram_V
        



    '''get Vocabulary from training TAG set. not including UNK
    :param: input tokenized training tags, type: list 
    :output: unigrams {unigrams:count}
    '''
    def unigram_tag(self,train_token):
            # this is the total length/num of tokens in training data      
            self.total_words_len_tag = len(train_token)
            # initialize word_count pairs
            unigram_V = {}
            # initial work-count dict population
            for token in train_token:
                unigram_V[token]= unigram_V.get(token,0) + 1   
            return unigram_V
        


    '''ngram generator, n>1
    : param: input tokened texts with STOP sign, and UNK replaced
                could be either training data or test data or sentences
            START added
    '''
    def ngrams_gen(self,tokens, n):
        #start_time = time.time()
        ngrams_tp = tuple()
        text = ' '.join(tokens)
        text = text.replace(STOP_token,STOP_token+'\n')
        
        sentences = set([w for w in text.splitlines()])

        for word in sentences:
            if(n == 2):
                word = START_token+' '+word
            if(n == 3):
                word = START_token+' '+START_token+' '+word
            it = iter(word.split())
            window = deque(islice(it, n), maxlen=n)
            yield tuple(window)       
            for item in it:
                window.append(item)
                yield tuple(window)
        ngrams_tp += tuple(window)
        yield ngrams_tp



    ''' 
        only for n=2,3 or more, generate {words:count}
        usually take training ngram tokens such as {a,b,c} as input, 
        generate ngram count with UNK
        when input test/dev data, it is used for error analysis
    
    ''' 
    def word_freq(self,tokens):
        #start_time= time.time()
        ngram_freq = {}
        # initial work-count dict population
        for token in tokens:       
            ngram_freq[token] = ngram_freq.get(token,0) + 1
        return ngram_freq
        
    

    ''' calculate MLE Probablity of unigram
        input word-freq dict for training data, which is Vocaborary
        this function will run even n specified by the shell is not 1
        traingin unigram
    '''
    def unigrams_prob(self,uni_count_dict):
        # probability dict {word:prob}
        prob_dict = uni_count_dict
        #print vocabulary
        items = prob_dict.iteritems() 
        for word, count in items:
            #print(count,total_words_len)
            #print('\n')
            prob_dict[word] = float(count) / float(self.total_words_len_tag)
        #print("--- %s seconds for unigrams_prob ---" % (time.time() - start_time))
        return prob_dict




    '''
    calculate MLE probability of ngram, n>=2
    : param: n: count dict of ngram,start from bigram
    : param: input untokened train texts with STOP sign
    trainign ngram. only use training data
    '''
    def ngram_prob(self,n):
        #print('------start ngram_prob---------------')
        start_time = time.time()
        # generate {ngrams:count} from training data
        ngram_list = list(self.ngrams_gen(self.train_tag_tokens,n))
        
        ngram_count_pairs = self.word_freq(ngram_list)
        prob_dict = ngram_count_pairs
        
        #print prob_dict
        if(n == 2):
            items = prob_dict.iteritems()     
            uni_count = self.unigram_count.copy()
            # add start token
            uni_count[START_token] = uni_count[STOP_token]
            # current probablity and word, in case n = 2, input is bigram words:count dict
            # input {a,b}: count, continue to get {a}: count
            for words, count in items:
                # extract the first item in bigram. 
                prior_word = words[0]   
                # get the count from {unigram: count} generated before       
                cnt_prior = uni_count[prior_word]    
                # q(w/v) = c(v,w)/c(v)      
                prob_dict[words] = count / cnt_prior
            # this should save as global for later use as bigram_prob_dict
            return prob_dict
        if(n == 3):
            items = prob_dict.iteritems() 
            # get {n-1gram:count} pairs
            priorgram_list = list(self.ngrams_gen(self.train_tag_tokens,n-1))
            priorgram_count_pairs = self.word_freq(priorgram_list)
            priorgram_count_pairs[START_token,START_token] = self.unigram_count[STOP_token]
            #-----------need to discard first few items--------
            for words, count in items:
                prior_word = words[:n-1]
                cnt_prior = priorgram_count_pairs[prior_word]
                #print(prior_word,words,cnt_prior,count)
                prob_dict[words] = count / cnt_prior
            return prob_dict
    # print("--- %s seconds for ngram_prob ---" % (time.time() - start_time))
                

    ##
    ##Evaluate the (negative) log probability of this word in this context.
    ##:param word: the word to get the probability of
    #:param prob_dict: the context the word is in
    ##
    def logprob(self,word,prob_dict):
        prob_dict = prob_dict
        return -math.log(prob_dict[word], 2)
        

    '''
    add k smoothing for unigram,trigram and bigram, similar to ngrams_prob()
    input training data and dev data
    return add k {ngram: prob} for dev/test data
    '''
    def add_k_smoothing(self,n,k):
        # generate {ngrams:count} from training data
        # print('------start add_k_smoothing---------------')
        text = self.dev_text
        tokens = self.tokenize_unigram(text)
        # number of words in text
        #text_len = len(tokens)   
        #global vocabulary
        
        sentences = set([s for s in text.splitlines()])
        # number of sentences
        #sent_num = len(sentences)
        #voc_set = set(prob_dict.keys())
        new_prob_dict = {}
    
        if (n ==1):      
            prob_dict = self.unigram_count
            voc_set = set(prob_dict.keys())
            for sent in sentences:
                sent_temp = self.tokenize_unigram(sent)
                for word in sent_temp:
                    if word not in voc_set:
                        #entr += self.logprob(UNK_token, prob_dict)
                        new_prob_dict[word] = (float(prob_dict[UNK_token])+k)/ (float(self.total_words_len_tag)+self.V*k)
                        #prob_dict[word] = (float(count)+k)/ (float(self.total_words_len)+self.V*k)
                    else:
                        #entr += self.logprob(word, prob_dict)
                        new_prob_dict[word] = (float(prob_dict[word])+k)/ (float(self.total_words_len_tag)+self.V*k)
            return new_prob_dict
        if(n > 1):   
        
            # training ngram
            ngram_list = list(self.ngrams_gen(self.train_tag_tokens,n))
            ngram_count_pairs = self.word_freq(ngram_list)
            prob_dict = ngram_count_pairs
            # voc_set in training bigram
            voc_set = set(prob_dict.keys())
            #uni_count = self.unigram_count

            # training n-1 gram
            prior_ngram_list = list(self.ngrams_gen(self.train_tag_tokens,n-1))
            prior_ngram_count_pairs = self.word_freq(prior_ngram_list)
            prior_prob_dict = prior_ngram_count_pairs
            # voc_set in training bigram
            prior_voc_set = set(prior_prob_dict.keys())

            for sent in sentences:
                # generate ngram for single sentence test data
                ngram_tmp = tuple(self.ngrams_gen(self.tokenize_unigram(sent), n))
                #      print ngram_tmp
                #print type(ngram_tmp)
                # iterate ngram in one sentence, skip first n-1 items
                for i in xrange(n - 1, len(list(ngram_tmp))):
                    #print i, ngram_tmp[i]
                    words = ngram_tmp[i]
                    prior_word = words[:n-1]
                    if words not in voc_set:
                        if prior_word not in prior_voc_set:
                            #new_prob_dict[words] = (float(prob_dict[UNK_token])+k)/ (float(self.total_words_len)+self.V*k)
                            new_prob_dict[words] = 1/self.V
                        else:
                            new_prob_dict[words] = (k)/ (prior_prob_dict[prior_word]+self.V*k)
                    else:
                         new_prob_dict[words] = (prob_dict[words] + k)/ (prior_prob_dict[prior_word]+self.V*k)
            return new_prob_dict
                  
           

    '''
    linear interpolation trigram, use ngrams_prob()
    output probability dict of test/dev data
    take training data and test/dev data
    '''
    def trigram_linear_interpolation(self,la1,la2,la3):
        # entr = 0.0
        n = 3
        new_prob_dict = {}
        bigram_p_dict = self.bigram_prob_dict.copy()
        trigram_p_dict = self.trigram_prob_dict.copy()
        unigram_p_dict = self.unigrams_prob_dict.copy()

        bi_keys_set = set(bigram_p_dict.keys())
        uni_keys_set = set(unigram_p_dict.keys())
        tri_keys_set = set(trigram_p_dict.keys())

        tag_space = self.tag_space
        N = len(tag_space)
        # add (START,START,tag) 
        for n in xrange(0,N):
            first_tri = START_token,START_token,tag_space[n]
            biword = first_tri[1:]
            if first_tri not in tri_keys_set:
                trigram_p_dict[first_tri]=0
            if biword not in bi_keys_set:
                bigram_p_dict[biword] = 0

            new_prob_dict[first_tri] = float(la1) * trigram_p_dict[first_tri] +\
                    float(la2) * bigram_p_dict[START_token,tag_space[n]] + float(la3) * unigram_p_dict[tag_space[n]]

            # add (START, tag, STOP)
            special_tri = START_token,tag_space[n],STOP_token
            biword3 = special_tri[1:]
            if special_tri not in tri_keys_set:
                trigram_p_dict[special_tri]=0
            if biword3 not in bi_keys_set:
                bigram_p_dict[biword3] = 0

            new_prob_dict[special_tri] = float(la1) * trigram_p_dict[special_tri] +\
                    float(la2) * bigram_p_dict[biword3] + float(la3) * unigram_p_dict[STOP_token]


        # add (tag,tag,STOP_token) and (START, tag, tag)
        for i in xrange(0,N):
            for j in xrange(0,N):
                sec_tri = START_token,tag_space[i],tag_space[j]
                last_tri = tag_space[i],tag_space[j],STOP_token
                biword1 = sec_tri[1:]
                biword2 = last_tri[1:]
                if biword1 not in bi_keys_set:
                    bigram_p_dict[biword1]=0
                if biword2 not in bi_keys_set:
                    bigram_p_dict[biword2]=0
                if sec_tri not in tri_keys_set:
                    trigram_p_dict[sec_tri]=0
                if last_tri not in tri_keys_set:
                    trigram_p_dict[last_tri]=0
                new_prob_dict[sec_tri] = float(la1) * trigram_p_dict[sec_tri] +\
                    float(la2) * bigram_p_dict[biword1] + float(la3) * unigram_p_dict[tag_space[j]]
                new_prob_dict[last_tri] = float(la1) * trigram_p_dict[last_tri] +\
                    float(la2) * bigram_p_dict[biword2] + float(la3) * unigram_p_dict[STOP_token]


        for i in xrange(0,N):
            for j in xrange(0,N):
                for k in xrange(0,N):
                    triword = tag_space[i],tag_space[j],tag_space[k]
                    bi_word = triword[1:]
                    uniword = triword[2]
                    if bi_word not in bi_keys_set:
                        bigram_p_dict[bi_word] = 0
                    if triword not in tri_keys_set:
                        trigram_p_dict[triword]=0
                    new_prob_dict[triword] = float(la1) * trigram_p_dict[triword] +\
                    float(la2) * bigram_p_dict[bi_word] + float(la3) * unigram_p_dict[uniword]

        #print('len of tri_prob_dict',len(new_prob_dict))
        return new_prob_dict




        '''
    linear interpolation bigram, use ngrams_prob()
    take train tokens and get transition probabilty for every pair in
    viterbi trellis
    input tag pairs in tag-space
    output numpy array of transistion prob

    '''
    def bigram_linear_interpolation(self,la1,la2):
        #construct every pair in tag space
        bigram_p_dict = self.bigram_prob_dict.copy()
        unigram_p_dict = self.unigrams_prob_dict.copy()
        bi_keys_set = set(bigram_p_dict.keys())
        uni_keys_set = set(unigram_p_dict.keys())
        tag_space = self.tag_space
        N = len(tag_space)
        # bi_trellis[0][i] is START: tagi
        # bi_trellis[N+1][i] is STOP: tagi
        # bi_trellis[i+1][i] is tag:tag.. e.g.  [1][0] is tag 1: tag 1
        bi_trellis = np.empty((N+2,N))
        # filling start and stop across rows
        # iterate columns
        for i in xrange(0,N):
            biword1 = START_token,tag_space[i]
            biword2 = tag_space[i],STOP_token
            if biword1 not in bi_keys_set:
                bigram_p_dict[biword1] = 0
            if biword2 not in bi_keys_set:
                bigram_p_dict[biword2] = 0
            # bi_trellis[0][i]: transition prob from start to tag i
            bi_trellis[0][i] = float(la1) * bigram_p_dict[biword1] + float(la2) * unigram_p_dict[tag_space[i]]
            bi_trellis[N+1][i] = float(la1) * bigram_p_dict[biword2] + float(la2) * unigram_p_dict[STOP_token]
            #filling non stop or start. Iterate over rows
            for j in xrange(0,N):
                biword = tag_space[i],tag_space[j]
                if biword not in bi_keys_set:
                    bigram_p_dict[biword] = 0
                bi_trellis[j+1][i] = float(la1) * bigram_p_dict[biword] + float(la2) * unigram_p_dict[tag_space[j]]           
        # number of tokens in trellis
        #print('bi_trellis shape',bi_trellis.shape )
        
        return bi_trellis


    '''
    emit e(dog|N) = c(N~dog) + k/ c(N)
    input training data
    need to handle OOV and #,@, emoji
    the emssion of unseen words in dev/test will use 
    UNK count in training instead
    '''
    def emission_prob(self):
        emit_prob  = dict()   
                # check numbers, mention, RT.....   
        test_twt = self.test_tweets
        #count = 0
        tags = set(self.tag_space)
        for tag in tags:
            for sent in test_twt:
                sent_token = list()
                for word in sent:
                    word_text = word[0]
                    #self.unigram_count
                    if word_text in self.vocabulary_text:
                        emit_prob[word_text,tag] = self.replaced_pair_count[(word_text,tag)]/self.unigram_count[tag]
                    else:
                        if bool(patterns.NUMBERS_PATTERN.match(word_text)):
                            emit_prob[word_text,tag] = self.replaced_pair_count[(NUMBERS_token,tag)]/self.unigram_count[tag]            
                        elif bool(patterns.URL_PATTERN.match(word_text)):
                            emit_prob[word_text,tag] = self.replaced_pair_count[(URL_token,tag)]/self.unigram_count[tag]
                                
                        elif bool(patterns.EMOJIS_PATTERN.match(word_text)):
                            emit_prob[word_text,tag] = self.replaced_pair_count[(EMOJIS_token,tag)]/self.unigram_count[tag]
                                
                        elif bool(patterns.RT_PATTERN.match(word_text)):
                            emit_prob[word_text,tag] = self.replaced_pair_count[(RT_token,tag)]/self.unigram_count[tag]
                                
                        elif bool(patterns.HASHTAG_PATTERN.match(word_text)):
                            emit_prob[word_text,tag] = self.replaced_pair_count[(HASHTAG_token,tag)]/self.unigram_count[tag]
                        elif bool(patterns.MENTION_PATTERN.match(word_text)):
                            emit_prob[word_text,tag] = self.replaced_pair_count[(MENTION_token,tag)]/self.unigram_count[tag]
                        else: 
                            emit_prob[word_text,tag] = self.replaced_pair_count[(UNK_token,tag)]/self.unigram_count[tag]
        #handle special chararcers, #, @, emoji     
        return emit_prob

    '''
    viterbi for bigram
    input: tag space of lenth N: vocabulary_tag
          sentences of lenth T: list of tokens
           trellis: 2d array of trained add-k or liner interpolation bigram
    return: best path(best tag sequence, or estimation tages) 
           in list
    '''
    def viterbi_bi(self,sent_tokens,trellis):
        # probability matrix viterbi[N+1,T], including stop
        # backpointer path matric[N+1,T]
        # T observations
        T = len(sent_tokens)
        # N states/tags not including stop and start     
        N = len(self.tag_space)
        # transition prob
        trellis = trellis
        prob_mat = np.empty((N+1,T))
        #path_mat = np.empty((N+1,T))
        tag_space = self.tag_space
        # a dict of strings of tags
        path= {}
        predicted_tags = list()
        # initialization, base case
        # assume bigram first, ignore start
        # trans
        for i in xrange(0,N):
            # from start to the first tag * emission from first word 
            # conditions on first tag
            prob_mat[i][0] = trellis[0][i]*self.emit_prob[sent_tokens[0],tag_space[i]]
            path[tag_space[i]] = tag_space[i]
        for t in xrange(1,T):
            newpath = {}
            # iterate through states
            for n in xrange(0,N):
                # previous transition tag n0
                prob_mat[n][t], state=max(((prob_mat[n0][t-1])*trellis[n0+1][n]* self.emit_prob[sent_tokens[t],tag_space[n]],n0) for n0 in xrange(0,N))
                #path_mat[n][t]= tag_space[n0]
                #print('T',t,'N',n)
                #print(prob_mat[n][t])
                newpath[tag_space[n]] = path[tag_space[state]]+tag_space[n]
            path = newpath                
        # terminaiton prob_mat[N][T-1] = STOP token:last observation
        prob_mat[N][T-1],state = max(((prob_mat[n0][T-1]*trellis[N+1][n0]),n0)for n0 in xrange(0,N))
        #final_path = path[tag_space[state]]
        #path_mat[N][T-1] = 
        predicted_tags = list(path[tag_space[state]])
        #print(predicted_tags)
        return predicted_tags


    '''
    take dev/test data and produce estimated tags
    n: trigram or bigram viterbi
    calculate accuracy and confusion matrx
    '''
    def test_hmm(self,trellis,n):
        text_token = list()
        tag_token = list()
        estimated_token = list()
        test_twt = self.test_tweets
        count = 0
        for sent in test_twt:
            sent_token = list()
            for word in sent:
                text_token.append(word[0])
                tag_token.append(word[1])
                sent_token.append(word[0])
                    # do bi_viterbi
            if(n == 2):
                estimated_token.append(self.viterbi_bi(sent_token,trellis))
            if(n==3):
                start_time = time.time()
                estimated_token.append(self.viterbi_tri(sent_token,trellis))
                end_time = time.time()
                count += 1
                #print('-----------------------------------')
                #print('count: ' + str(count) + ' time: ' + str(end_time-start_time))
        # flatten list   
        estimated_token = [item for sublist in estimated_token for item in sublist]
            # accuracy_score(y_true, y_pred)        
        accuracy = accuracy_score(tag_token, estimated_token) 
        cnf_matrix = confusion_matrix(tag_token, estimated_token)
        return accuracy,cnf_matrix




    def viterbi_tri(self,sent,trigram):
            pi = {}
            bp_mat_prev = {}
            tag_vocab = set(self.tag_space)
            # initialization
            pi[0,START_token,START_token] = 1
            bp_mat_prev[START_token,START_token] = []
            n = len(sent)
            for k in range(1,n+1):
                    word = sent[k-1]
                    bp_mat = {}
                    #w, u, v: tag of k-2, k-1, k-th word
                    for u in self.getTagVocab(k-1, tag_vocab):
                            for v in self.getTagVocab(k, tag_vocab):
                                    e = self.emit_prob[word,v]
                                    pi[k,u,v], bp_w = max([(e*trigram[w,u,v]*pi[k-1,w,u], w) for w in self.getTagVocab(k-2, tag_vocab)])
                                    bp_mat[u,v] = bp_mat_prev[bp_w,u] + [v]
                    bp_mat_prev = bp_mat

            # yn, ym (m = n-1)
            score, ym, yn = max([(pi[n,ym,yn] * trigram[ym,yn,STOP_token], ym, yn) for ym in self.getTagVocab(n-1,tag_vocab) for yn in self.getTagVocab(n, tag_vocab)])
            #print bp_mat_prev[ym,yn]
            return bp_mat_prev[ym,yn]

    
    def getTagVocab(self,k, tag_vocab):
            if (k == -1 or k == 0):
                return set([START_token])
            else:
                return tag_vocab




    def plot_confusion_matrix(self, cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],fontsize=7,
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('hmm_trigram.png')
        



    '''
    initializaiton HMM
    '''
    def initHMM(self):
       
        self.train_tweets = self.json_read(self.training_set)
        self.test_tweets = self.json_read(self.dev_set)
        #print(self.train_tweets[:4])
        self.train_text_tokens, self.train_tag_tokens = self.json_process(self.train_tweets)
        self.test_text_tokens, self.test_tag_tokens = self.json_process(self.test_tweets)
        print('read json done!')

        self.unigram_count = self.unigram_tag(self.train_tag_tokens)
        #print("--- %s seconds for unigram count ---" % (time.time() - start_time))
        #print(self.unigram_count)
        # a list of vocabulary in unigrams
        self.vocabulary_tag = set(self.unigram_count.keys())

        # generate tweet tag pair and replace UNK_token
        unigram_text = self.unigram_V(self.train_text_tokens)
        self.vocabulary_text = set(unigram_text.keys())
        
        # generate unigram probablity dict
        uni_prob_dict = {}
        uni_prob_dict = self.unigram_count.copy()
        self.unigrams_prob_dict = self.unigrams_prob(uni_prob_dict)
        #print("--- %s seconds for unigram prob ---" % (time.time() - start_time))
        # tag space V
        self.V = len(self.vocabulary_tag)
        # print("Vocabulary lenth",self.V)
        # print('total_words_len',self.total_words_len_tag)
    
        print("training unigram finished")
        self.tag_space = list(self.unigram_count.keys())
        self.tag_space.remove(STOP_token)
        print('tag space:')
        print self.tag_space

        #       # generate trigram probability dict
        self.trigram_prob_dict = self.ngram_prob(3)
        print("training trigram finished")
        #print(self.trigram_prob_dict)
        # generate bigram probability dict
        self.bigram_prob_dict = self.ngram_prob(2)
        print("training bigram finished")

        self.emit_prob = self.emission_prob()

       
        # print('------bigram HMM on development data-----')

        # la_ls = [(0.001,0.999),(0.2,0.8),(0.5,0.5),(0.8,0.2),(0.999,0.001)]

        # for lamda in la_ls:
        #     print('lamda1,lamda2:',lamda[0],lamda[1])
        #     bi_trellis = self.bigram_linear_interpolation(lamda[0],lamda[1])
        #     accuracy, confusion_matrix = self.test_hmm(bi_trellis,2)
        #     print('accuracy',accuracy)
           
        # print('\n')

      
        print('------bigram HMM on test data-----')
        lamda = (0.001,0.999)
        print(lamda)
        bi_trellis = self.bigram_linear_interpolation(lamda[0],lamda[1])
        accuracy, confusion_matrix = self.test_hmm(bi_trellis,2)
        print('accuracy',accuracy)
       
       
        #print(confusion_matrix)
        # classes = self.tag_space
        # self.plot_confusion_matrix(confusion_matrix, classes=classes,normalize=False,
        #               title='Confusion matrix, bigram HMM')

      
 

        #print('------trigram HMM on development data-----')

        # la_ls = [(0.001,0.009,0.99),(0.3,0.3,0.4),(0.6,0.3,0.1),(0.99,0.009,0.001)]

        # for lamda in la_ls:
        #     print('lamda1,lamda2,lamda3:',lamda[0],lamda[1],lamda[2])
        #     tri_li_prob_dict = self.trigram_linear_interpolation(lamda[0],lamda[1],lamda[2])
        #     accuracy, confusion_matrix = self.test_hmm(tri_li_prob_dict,3)
        #     print('accuracy',accuracy)
           
        # print('\n')


        print('------trigram HMM on test data-----')
        lamda = (0.6,0.3,0.1)
        tri_li_prob_dict = self.trigram_linear_interpolation(lamda[0],lamda[1],lamda[2])
        accuracy, confusion_matrix = self.test_hmm(tri_li_prob_dict,3)
        print('accuracy',accuracy)
       
       
        print(confusion_matrix)
        classes = self.tag_space
        self.plot_confusion_matrix(confusion_matrix, classes=classes,normalize=False,
                      title='Confusion matrix, trigram HMM')   
        self.plot_confusion_matrix(confusion_matrix, classes=classes, normalize=True,
                   title='Normalized confusion matrix, trigram HMM')
        
       

'''
 input n, test and training, right now no smoothing 
 output perplexity
'''
def main():
  
    args = get_args()
    hmm = HMM(args)
    hmm.initHMM()
       
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_set", action="store",
                        help="training data.")
    parser.add_argument("dev_set", action="store",
                        help="either dev data or test data. dev for tunning, test can be used only once")
    parser.usage = ("yanan_lm.py [-h] [-n N] training_set dev_set")
    parser.add_argument("-t", "--threshold", action="store", type=int,
                        default=1, metavar='T',
                        help="threshold value for words to be UNK.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
