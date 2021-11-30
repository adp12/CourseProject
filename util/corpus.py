import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from dateutil import parser as dparser
import urllib.parse

import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer



def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

class Corpus(object):
    
    #******************************************************************************
    #----------------------------------Method---------------------------------------
    #******************************************************************************
    
    #Run the web_query to produce a collection of text documents scraped from the web
    
    #Use the set_results() function to store the full results in the corpus for processing
    
    #Use the set_corpus() function to assign the documents scraped from the web to the corpus
    
    #Sub divide the documents into smaller sub_docs
    
    #Rank the documents based on relevance to the original query as well as any tags
    
    #Prune the sub_docs to produce a relevant set
    
    #
    
    #******************************************************************************
    #******************************************************************************
    
    def __init__(self):
        
        #typical corpus data
        self.documents = []
        self.vocabulary = []
        self.number_of_documents = 0
        self.vocabulary_size = 0
                
        #for web results
        self.query_results=None
        self.max_tokens=512
        self.failed = []
        
        #sub dividing documents
        self.tokenizer=None
        self.sub_docs=None
        self.sub_list=[]
        
        #queries
        self.prime_q=None
        self.expanded_q=None        
        
        #relevance scores
        self.document_scores=None
        self.subdoc_scores=None
        self.title_scores=None
        self.sub_list_scores=None
                
        #pruned data
        self.pruned_docs=[]
        self.pruned_subdocs={}
        
        #relevant
        self.relevant_set={}
        self.relevant_scores={}
        self.rel_list_scores=[]
        
        #sentiments
        self.sentiments=None
    
    
    #******************************************************************************
    #------------------------------Setting Corpus----------------------------------
    #******************************************************************************
    
    def set_results(self, df):
        #dataframe returned from webquery
        self.query_results=df
    
    def set_corpus(self, documents):
        self.documents = documents
                

    def build_corpus_from_file(self, file_path):

        f = open(file_path, 'r')
        docs = f.readlines()
        for d in docs:
            self.documents.append(d)
        self.number_of_documents = len(docs)

        
    def build_vocabulary(self, stopwords):

        v = set([])
        for x in self.documents:
            tmp = set(x.split())
            tmp = {x for x in tmp if x.lower() not in stopwords}
                        
            v.update(tmp)
        
        v = list(v)
        self.vocabulary = v
        self.vocabulary_size = len(v)
    
    def build_queries(self, ticker, prime_w=1, tag_w=0.05, include_gtags=True, gtags_w=0.01, include_btags=True, btags_w=0.01):
        name = re.sub('(,|\.|Inc|inc|company|co )',"",str(ticker.name))
        self.prime_q = name
        
        gtags = ['investing','analysis','analyst','upgrade','downgrade']
        btags = ['sentiment','opinion','outlook']
        
        exp = []
        for t in ticker.tags:
            exp.append([t, tag_w])
        if include_gtags:
            for t in gtags:
                exp.append([t, gtags_w])
        if include_btags:
            for t in btags:
                exp.append([t, btags_w])
        
        self.expanded_q=exp
    
    #******************************************************************************
    #------------------------------Sub Dividing-------------------------------------
    #******************************************************************************
    
    def get_pgraphs(self, doc, cutoff, method):
        #updated get_pgraphs() with method for cutoff
        #cut off method:
        #sen: number of sentences
        #word: number of words  

        pgraphs=[]
        freshsoup = re.split('\n\n',doc)
        for x in range(0,len(freshsoup)):
            if method=='word':
                words = len(str(freshsoup[x]).strip().split(' ',maxsplit=cutoff))
                if words>cutoff:
                    pgraphs.append(freshsoup[x])
                    
            elif method=='sen':
                sens = len(re.findall("\.",str(freshsoup[x]).strip()))
                if sens>cutoff:
                    pgraphs.append(freshsoup[x])
                    
        return pgraphs
    
    def split_doc(self, doc):         
        
        if len(doc)>0:
            avgwrds = 15
            estsens = len(doc)/avgwrds
        
            if len(re.findall('\.',doc))>=estsens:
                cut_point = doc.rfind('.', 0, int(len(doc)/2))
                if cut_point<=0:
                    cut_point = int(len(doc)/2)
            else:
                cut_point = int(len(doc)/2)

            d1 = doc[0:cut_point]
            d2 = doc[cut_point+1:]
            
            tkns1 = int(len(self.tokenizer(d1)['input_ids']))

            if tkns1>self.max_tokens:
                self.split_doc(d1)
            else:
                if len(d1)>0:
                    self.subs.append(d1)

            tkns2 = int(len(self.tokenizer(d2)['input_ids']))

            if tkns2>self.max_tokens:
                self.split_doc(d2)
            else:
                if len(d2)>0:
                    self.subs.append(d2)

    
    def get_subdocs(self, pgraphs):
        #Updated get_subdocs with iterative slicing 
        #Changed to recursive slicing
        #ensure sub_docs tokens will not exceed max_tokens for sentiment model
        self.subs=[]

        for x in range(0, len(pgraphs)):
            sen_cnt = len(re.split('\n|\. ',pgraphs[x]))
            tkns = int(len(tokenizer(pgraphs[x])['input_ids']))

            if tkns<self.max_tokens:
                self.subs.append(pgraphs[x])
            else:
                self.split_doc(pgraphs[x])
        
        return self.subs
        
    def sub_divide(self, tokenizer, cutoff=1, method='sen'):

        #creates a dictionary of sub_docs divided from each document in the corpus
        #method: using get_pgraphs() followed by get_subdocs()
        #output form: dict{ document_id : [subdoc_1, subdoc_2 ... subdoc_n] }

        subbed_data = {}
        self.tokenizer=tokenizer
        
        if len(self.pruned_docs)==0:
            self.prune_docs()
        
        for x in range(0, len(self.documents)):
            
            #only include documents that made the first relevance cut
            if x in self.pruned_docs:
                pg = self.get_pgraphs(self.documents[x], cutoff, method)
                subs = self.get_subdocs(pg)
                subbed_data[x]=subs
        
        self.sub_docs = subbed_data
        self.sub_list=[]
        for x in self.sub_docs.keys():
            for y in self.sub_docs[x]:
                self.sub_list.append(y)
        

        
    #******************************************************************************
    #----------------------------------Relevance Scoring---------------------------
    #******************************************************************************  
    
    def rank_docs(self, ranker):
        query = self.prime_q
        self.document_scores = ranker.score(query, self.documents)
        
        
    def rank_subdocs(self, ranker, expanded=False):
        sub_vecs={}
        
        if expanded==False:
            query=self.prime_q
            for x in self.sub_docs.keys():
                sub_vec = ranker.score(query, self.sub_docs[x])
                sub_vecs[x]=sub_vec
        else:
            query = self.expanded_q
            for x in self.sub_docs.keys():
                sub_vec = ranker.score_expanded(query, self.sub_docs[x])
                sub_vecs[x]=sub_vec
            
        self.subdoc_scores = sub_vecs
        
        self.sub_list_scores=[]
        for x in self.subdoc_scores.keys():
            for y in self.subdoc_scores[x]:
                self.sub_list_scores.append(y)
    
    def rank_relevant(self, ranker, expanded=True):
        sub_vecs={}
        
        if expanded==False:
            query=self.prime_q
            for x in self.relevant_set.keys():
                sub_vec = ranker.score(query, self.relevant_set[x])
                sub_vecs[x]=sub_vec
        else:
            query=self.expanded_q
            for x in self.relevant_set.keys():
                sub_vec = ranker.score_expanded(query, self.relevant_set[x])
                sub_vecs[x]=sub_vec
            
        self.relevant_scores = sub_vecs
        
        self.rel_list_scores=[]
        for x in self.relevant_scores.keys():
            for y in self.relevant_scores[x]:
                self.rel_list_scores.append(y)
       
        
    #******************************************************************************
    #----------------------------Pruning Relevant Set------------------------------
    #******************************************************************************
    def prune_docs(self, method='finite', cutoff=0):
        
        '''
        method percentile: cutoff is the percentile to lower bound the document scores on
        method finite: cutoff is a hard value to cutoff scores on
        
        store the indexes of the documents that have scores over the cutoff
        these indexes will be used in the creation of the subdocs
        '''
        
        if method=='percentile':
            p = np.percentile(self.document_scores, cutoff)
            cuts = np.where(self.document_scores<p)
            
            for x in range(0, len(self.documents)):
                if x not in cuts:
                    self.pruned_docs.append(x)
                                                 
        elif method=='finite':
            for x in range(0, len(self.documents)):
                if self.document_scores[x]>cutoff:
                    self.pruned_docs.append(x)
                    
            
    def prune_subdocs(self, method='finite', cutoff=0):
        
        '''
        method percentile: cutoff is the percentile to lower bound the document scores on
        method finite: cutoff is a hard value to cutoff scores on
        
        '''
        
        if method=='percentile':
            p = np.percentile(self.sub_list_scores, cutoff)
            prune={}
            for x in self.sub_docs.keys():
                for y in range(0, len(sub_docs[x])):
                    if subdoc_scores[x][y]>p:
                        if x not in prune.keys():
                            prune[x]=[y]
                        else:
                            prune[x].append(y)
            self.pruned_subdocs=prune
            
        elif method=='finite':
            prune={}
            for x in self.sub_docs.keys():
                for y in range(0, len(self.sub_docs[x])):
                    if self.subdoc_scores[x][y]>cutoff:
                        if x not in prune.keys():
                            prune[x]=[y]
                        else:
                            prune[x].append(y)
            
            self.pruned_subdocs=prune
        
    def make_relevant(self):
        #Create the relevant_set from the sub_doc keys in the pruned_subdocs dict
        
        for x in self.pruned_subdocs.keys():
            self.relevant_set[x]=[]
            self.relevant_scores[x]=[]
            for y in self.pruned_subdocs[x]:
                self.relevant_set[x].append(self.sub_docs[x][y])
                self.relevant_scores[x].append(self.subdoc_scores[x][y])
        
        self.rel_list=[]
        for x in self.relevant_set.keys():
            for y in self.relevant_set[x]:
                self.rel_list.append(y)
        
    def prune_relevant(self, method='percentile',cutoff=15):
        #Relevant is as low as it goes, these will be adjusted directly when pruned
        
        if method=='percentile':
            cut = np.percentile(self.rel_list_scores, cutoff)
        else:
            cut = cutoff
        
        subbed_data = self.relevant_set
        sub_scores = self.relevant_scores
        
        for x in self.relevant_set.keys():

            subbed_data[x] = [xv if c else None for c, xv in zip(sub_scores[x]>cut, subbed_data[x])]
            subbed_data[x] = [y for y in subbed_data[x] if y!=None]
            sub_scores[x] = [y for y in sub_scores[x] if y>cut]
        
        self.relevant_set = {k: v for k, v in subbed_data.items() if len(v) > 0}
        self.relevant_scores={k: v for k, v in sub_scores.items() if len(v) > 0}
        
        self.rel_list=[]
        for x in self.relevant_set.keys():
            for y in self.relevant_set[x]:
                self.rel_list.append(y)
        
 
