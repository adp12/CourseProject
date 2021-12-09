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
import ipywidgets as widgets

import torch
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import pipeline

class Corpus(object):
    
    #******************************************************************************
    #----------------------------------Method---------------------------------------
    #******************************************************************************
    '''
    0.) Setup
        a.) Pull in api keys from config
        b.) create a Ticker() object
    
    1.) Getting web results and initiating Corpus class
        a.) Run the web_query to produce a collection of text documents scraped from the web
        b.) Use the set_results() function to store the full results in the corpus for processing (needed for dataset building)
        c.) Use set_corpus() with the web_query documents and urls
    
    2.) Setting up Ranker and Initial Queries
        a.) Initiate the ranking function (in our case, custom built BM25 object class from util.pyRanker)
        b.) Fit the ranker to the corpus documents
        c.) Build the queries used by the ranking function using build_queries() and passing in the ticker object from step 0

    3.) Initial Ranking
        a.) Rank corpus documents for relevance with rank_docs() passing in the ranker
        b.) Prune the documents with prune_docs(). This is pre-set to only prune 0 ranked documents on the primary query using a standard BM25 score
            b1.) Note: these documents are not removed, only indexed in the pruned_docs for the sub-division proecess
    
    4.) Sub-Dividing
        a.) Create dictionary of sub-docs from the original documents by calling sub_divide() and passing in the Transformer's tokenizer
            a1.) The tokenizer is needed to ensure that the length of the subdocs created will not exceed the maximum token size used by the Transformer
        b.) Rank the newly created subdocs using rank_subdocs() and pass in the ranker
        c.) Prune the subdocs with prune_subdocs(). This is pre-set similarly to prune_docs() and prunes 0 ranked subdocs using the prime query and standard BM25 score
            c1.) Like the prune_docs() function, this does not remove any subdocs, but creates an index of the ones to keep from the subdoc dictionary
    
    5.) Relevant Set
        a.) after sub-dividing and pruning out all the trash, make the relevant set by calling make_relevant()
        b.) Rank the relevant set with rank_relevant().
            b1.) This ranking function is pre-set to run a BM25 with Structured Query Expansion. The expanded query and its weights set can be adjusted when initiating build_queries()
        c.) If needed/wanted, you can further prune the relevant set using prune_relevant()
            c1.) Note: unlike the other two pruning functions, this directly adjusts the relevant_set and relevant_scores stored by the corpus
    
    6.) Sentiment
        a.) get_sentiments(): Once a relevant set is established
            a1.) passing in the Transformer's classifier will run each relevant subdoc through the classifier to produce a sentiment score
    
    7.) Dataset for graphing
        a.) initiate a data_manager object from util.data_manager.
            a1.) the data_manager() takes the '_data' directory on initiation
        b.) data_preprocess() will setup the needed dictionaries for creating pandas dataframes
        c.) build the two dataframes with build_fulldf() and build_pricedf()
        d.) tell the data_manager to put the data in a retrievable place with data_manager.store_date() and pass in the ticker symbol and dataframes
    
    
    '''


    #******************************************************************************
    #******************************************************************************
    
    def __init__(self, model=None, tokenizer=None):
        
        #corpus data
        self.documents = []
        self.number_of_documents = 0

        #for web results
        self.query_results=None
        self.urls=None
        self.failed = []
        
        #sub dividing documents
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
        
        #data set creation
        self.full_df=None
        self.price_df=None
        
        #Transformer
        if model==None:
            try:#Try to assign the model from our fine-tuned model for this project
                self.model = AutoModelForSequenceClassification.from_pretrained("adp12/cs410finetune1")
            except:#catch on error to assign model to standard distilBert
                self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        else:
            self.model=model
        
        if tokenizer==None:
            self.tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        else:
            self.tokenizer=tokenizer

        self.classifier = pipeline(task='sentiment-analysis',model=self.model,tokenizer=self.tokenizer)
        self.max_tokens = int(self.tokenizer.model_max_length)
        
        try:
            self.stopwords=[]
            with open('util/stopwords.txt') as f:
                self.stopwords.append(f.read().splitlines())
            self.stopwords=self.stopwords[0]
        except:
            print('Could not find stopwords file, make sure to assign in ranking function')
    
    
    #******************************************************************************
    #------------------------------Setting Corpus----------------------------------
    #******************************************************************************
    
    def set_results(self, df):
        #dataframe returned from webquery
        self.query_results=df
    
    def set_corpus(self, documents, urls):
        self.documents = documents
        self.urls = urls
                
    def build_corpus_from_file(self, file_path):

        f = open(file_path, 'r')
        docs = f.readlines()
        for d in docs:
            self.documents.append(d)
        self.number_of_documents = len(docs)
    
    
    #******************************************************************************
    #---------------------------------Queries--------------------------------------
    #******************************************************************************
    
    def set_ticker_data(self, ticker):
        name = re.sub('(Inc|Company|Corporation|Corp|Co|Incorporated|LLC|Group|Limited||,|\.)',"",str(ticker.name), flags=re.I)
        self.ticker_sym = str(ticker.ticker)
        self.company_name = str(name)
        self.ticker_tags = ticker.tags    
    
    def build_queries(self, ticker=None, prime='name', prime_w=1, tag_w=0.05, include_gtags=True, gtags_w=0.01, include_btags=True, btags_w=0.01):
        
        if ticker is not None:
            self.set_ticker_data(ticker)
        
        if prime=='name':
            self.prime_q = str(self.company_name)
            self.prime_type=1
        elif prime=='ticker':
            self.prime_q = str(self.ticker_sym)
            self.prime_type=2
        elif prime=='both':
            self.prime_q = str(self.company_name)+" "+str(self.ticker_sym)
        
        gtags = ['investing','analysis','analyst','upgrade','downgrade']
        btags = ['sentiment','opinion','outlook']
        
        exp = []
        exp.append([self.prime_q, prime_w])
        for t in self.ticker_tags:
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
            tkns = int(len(self.tokenizer(pgraphs[x])['input_ids']))

            if tkns<self.max_tokens:
                self.subs.append(pgraphs[x])
            else:
                self.split_doc(pgraphs[x])
        
        return self.subs
        
    def sub_divide(self, tokenizer=None, cutoff=1, method='sen'):

        #creates a dictionary of sub_docs divided from each document in the corpus
        #method: using get_pgraphs() followed by get_subdocs()
        #output form: dict{ document_id : [subdoc_1, subdoc_2 ... subdoc_n] }

        subbed_data = {}
        if tokenizer is not None:
            self.tokenizer=tokenizer
        
        if len(self.pruned_docs)==0:
            self.prune_docs()
        
        c = len(self.documents)
        pgres = widgets.IntProgress(value=0,min=0,max=c, step=1)
        display(pgres)
        
        for x in range(0, len(self.documents)):
            
            #only include documents that made the first relevance cut
            if x in self.pruned_docs:
                pg = self.get_pgraphs(self.documents[x], cutoff, method)
                subs = self.get_subdocs(pg)
                subbed_data[x]=subs
                
            pgres.value+=1
            pgres.description=str(pgres.value)+":"+str(c)
            
        self.sub_docs = subbed_data
        self.sub_list=[]
        for x in self.sub_docs.keys():
            for y in self.sub_docs[x]:
                self.sub_list.append(y)
        

        
    #******************************************************************************
    #----------------------------------Relevance Scoring---------------------------
    #******************************************************************************  
    
    def rank_docs(self, ranker, min_max=0.02):
        query = self.prime_q
        scr = ranker.score(query, self.documents)
        if np.max(scr)>min_max:
            self.document_scores = scr
        else:
            print('Very low relevance scores, attempting requery')
            #rebuilding queries with prime query being alternate of current prime(name or ticker)
            if self.prime_type==1:
                self.build_queries(prime='ticker')
            else:
                self.build_queries(prime='name')
        
        query = self.prime_q
        scr = ranker.score(query, self.documents)
        if np.max(scr)>min_max:
            self.document_scores = scr
        else:
            print('Scoring on built queries from ticker resulting in unusually low scores')
            print('Change query manually in rank_docs function or adjust min_max')
        
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
                for y in range(0, len(self.sub_docs[x])):
                    if self.subdoc_scores[x][y]>p:
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
    
    #******************************************************************************
    #--------------------------------Sentiment Scoring-----------------------------
    #******************************************************************************    
    
    def get_sentiments(self, classifier=None):
        
        if classifier is not None:
            self.classifier = classifier
            
        c = 0
        for x in self.relevant_set.keys():
            c+=len(self.relevant_set[x])

        pgres = widgets.IntProgress(value=0,min=0,max=c, step=1)
        display(pgres)

        self.sentiments = {}

        for x in self.relevant_set.keys():
            scrs=[]
            for y in range(0, len(self.relevant_set[x])):

                s = self.classifier(self.relevant_set[x][y])
                scr = s[0]['score']
                if s[0]['label']=="NEGATIVE":
                    scr=scr*-1
                scrs.append(scr)
                pgres.value+=1
                pgres.description=str(pgres.value)+":"+str(c)

            self.sentiments[x]=scrs
    
    #******************************************************************************
    #--------------------------------Dataset functions-----------------------------
    #******************************************************************************
    def data_preprocess(self):
        #Collecting dictionaries to be used as dataframe columns
        #Multiplying relevance and sentiment to get general relevance weighted sentiment score

        self.scores={}
        self.avg_scores={}
        self.avg_relevance={}
        self.rel_urls={}
        self.sub_count={}

        for x in self.relevant_scores.keys():

            rw = []
            #every row, mul(relevace and sentiment) and keep data structure
            for y in range(0, len(self.relevant_scores[x])):
                rw.append(self.relevant_scores[x][y] * self.sentiments[x][y])

            self.rel_urls[x]=self.urls[x]
            self.sub_count[x]=len(self.relevant_set[x])
            self.scores[x]=rw
            #averages (used in graphing)
            self.avg_relevance[x]=np.mean(self.relevant_scores[x])
            self.avg_scores[x]=np.mean(rw)

        #Getting maximum used relevant subdocs for full_data storage
        m = max(self.sub_count, key=self.sub_count.get)
        self.max_subs = self.sub_count[m]

        #Getting average sentiment scores per data row (aka dict key)
        self.avg_sentiments={}
        for x in self.sentiments:
            self.avg_sentiments[x]=np.mean(self.sentiments[x])
            
            
    def build_fulldf(self):
        #Building full dataframe, including average relevance, sentiments, scores and all the relevant subdoc 
        #Send to datamanager to store for evaluation and recalculation

        full_df = pd.DataFrame()
        full_df['url'] = pd.Series(self.rel_urls)
        full_df['scores'] = pd.Series(self.avg_scores)
        full_df['sentiments'] = pd.Series(self.avg_sentiments)
        full_df['relevance'] = pd.Series(self.avg_relevance)
        full_df['rel_subdocs'] = pd.Series(self.sub_count)
        full_df = full_df.dropna()

        #Joining the Url dates from the webquery df
        mergedf = self.query_results
        mergedf = mergedf[['url','pub_date']]
        full_df = full_df.join(mergedf.set_index('url'),on='url')

        #reordering
        full_df = full_df[['pub_date','url','scores','sentiments','relevance','rel_subdocs']]

        #Pandas to datetime
        #pd.to_datetime(newdf['pub_date'], format='%Y-%m-%dT%H:%M:%S')
        full_df.loc[:,'pub_date'] = pd.to_datetime(full_df.loc[:,'pub_date'], infer_datetime_format=True, utc=True)
        full_df.loc[:,'pub_date'] = pd.to_datetime(full_df.loc[:,'pub_date'].dt.strftime('%Y/%m/%d'))

        #Adding in columns for subdocs relevance and sentiments
        cols = ['sub_'+str(x)+'_rel' for x in range(1, self.max_subs+1)]
        rel_df = pd.DataFrame.from_dict(self.relevant_scores, orient='index', columns=cols)
        cols = ['sub_'+str(x)+'_sents' for x in range(1, self.max_subs+1)]
        sen_df = pd.DataFrame.from_dict(self.relevant_scores, orient='index', columns=cols)

        full_df = full_df.join(rel_df)
        full_df = full_df.join(sen_df)

        full_df = full_df.sort_values(by='pub_date')
        
        self.full_df = full_df
        
    def build_pricedf(self, ticker):
        if self.full_df is not None and len(self.full_df)>0:
            df = self.full_df[['pub_date','sentiments','relevance','scores']]
            df.loc[:,'pub_date'] = pd.to_datetime(df.loc[:,'pub_date'], infer_datetime_format=True, utc=True)
            df.loc[:,'pub_date'] = pd.to_datetime(df.loc[:,'pub_date'].dt.strftime('%Y/%m/%d'))

            #dfs grouped on date and average
            pt_sn = df.groupby(df['pub_date'])['sentiments'].mean()
            pt_rl = df.groupby(df['pub_date'])['relevance'].mean()
            pt_sc = df.groupby(df['pub_date'])['scores'].mean()
            #count number of sentiment scores to create used news volume
            pt_dt = df['sentiments'].groupby(df['pub_date']).count()

            #combine into single frame
            pt_sn = pt_sn.to_frame()
            pt_rl = pt_rl.to_frame()
            pt_sc = pt_sc.to_frame()
            pt_dt = pt_dt.to_frame().rename(columns={"sentiments":"Doc_Volume"})

            df = pd.concat([pt_rl, pt_sn, pt_sc, pt_dt], axis=1)
            df = df.reset_index()
            #get stock price for ticker
            tdf = ticker.get_price()
            #merge data with price
            price_df = pd.merge(df, tdf, left_on='pub_date', right_on='date')
            price_df = price_df.drop(columns=['date'])
            
            self.price_df = price_df
            