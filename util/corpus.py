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
        
        #plsa and liklihoods
        self.likelihoods = []
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
                
        #for web results
        self.query_results=None
        self.failed = []
        
        #sub dividing documents
        self.sub_docs=None
        
        #relevance scores
        self.document_scores=None
        self.document_tag_scores=None
        self.subdoc_scores=None
        self.subdoc_tag_scores=None
        self.title_scores=None
                
        #pruned data
        self.relevant_set=None
        self.relevant_scores=None
    
    
    #******************************************************************************
    #------------------------------Setting Corpus----------------------------------
    #******************************************************************************
    
    def set_results(self, df):
        #dataframe returned from webquery
        self.query_results=df
    
    def set_corpus(self, documents):
        self.documents = documents
        
    def build_corpus_from_url(self, max_docs=50):
        
        #scrape text from url-list to build corpus
        #(not recommended, use the same method from the web_query object and the set_corpus() method)
        
        url_list = self.query_results['url'].tolist()
        url_list = url_list[0:max_docs]
        
        pgres = widgets.IntProgress(value=0,min=0,max=len(url_list), step=1)
        display(pgres)
        
        failed=[]
        headers = {"User-Agent":"Mozilla/5.0"}
        for i in range(0,len(url_list)):
            try:
                response = requests.get(url=url_list[i],headers=headers)
                if response.status_code==200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    d = soup.get_text()
                    if len(d)>200:
                        self.documents.append(d)
                else:
                    self.failed.append(i)
            except:
                self.failed.append(i)

            finally:
                pgres.value+=1
                pgres.description=str(i+1)+":"+str(len(url_list))
                
        self.number_of_documents=len(self.documents)
        #remove failed url responses from dataset
        self.query_results = self.query_results.take(list(set(range(self.query_results.shape[0]))-set(self.failed)))
        

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
        
             
    
    #******************************************************************************
    #------------------------------Sub Dividing------------------------------------
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
    

    def get_subdocs(self, pgraphs, tokenizer, max_tokens=512):
        #Updated get_subdocs with iterative slicing 
        #ensure sub_docs tokens will not exceed max_tokens for sentiment model
        sub_docs=[]

        for x in range(0, len(pgraphs)):
            sen_cnt = len(re.split('\n|\. ',pgraphs[x]))
            tkns = int(len(tokenizer(pgraphs[x])['input_ids']))

            if tkns>=max_tokens:

                pg = pgraphs[x]
                slices=0

                while True:
                    #cut in half, count tokens
                    slices+=1
                    cut_point = pg.rfind(".",0,int(len(pg)/2))+1
                    cut_tkns = int(len(tokenizer(pg[0:cut_point])['input_ids']))    

                    if cut_tkns<max_tokens:
                        break
                    else:
                        #trim pg and recut, counting slices
                        pg = pg[0:cut_point]

                #loop through pgraph[x] using multiples of cutpoint to slice
                #append subdoc at each slice
                for i in range(0, (slices*2)):
                    pg = pgraphs[x][(cut_point*(i)):(cut_point*(i+1))]
                    sub_docs.append(pg)

            else:
                sub_docs.append(pgraphs[x])
        
        return sub_docs
        
    def sub_divide(self, tokenizer, cutoff=1, method='sen'):

        #creates a dictionary of sub_docs divided from each document in the corpus
        #method: using get_pgraphs() followed by get_subdocs()
        #output form: dict{ document_id : [subdoc_1, subdoc_2 ... subdoc_n] }

        subbed_data = {}

        for x in range(0, len(self.documents)):

            pg = self.get_pgraphs(self.documents[x], cutoff, method)
            subs = self.get_subdocs(pg, tokenizer)
            subbed_data[x]=subs

        self.sub_docs = subbed_data

        
    #******************************************************************************
    #---------------------------Relevance Scoring----------------------------------
    #******************************************************************************  
    
    def rank_docs(self, query, ranker):
        self.document_scores = ranker.score(query, self.documents)
        
    def rank_doc_tags(self, tags, ranker):
        tag_scores=[]
        for t in tags:
            scores = ranker.score(t, self.documents)
            tag_scores.append(scores)
            
        self.document_tag_scores = tag_scores
        
    def rank_subdocs(self, query, ranker):
        sub_vecs={}
        for x in self.sub_docs.keys():
            sub_vec = ranker.score(query, self.sub_docs[x])
            sub_vecs[x]=sub_vec
            
        self.subdoc_scores = sub_vecs
    
    def rank_subdocs_tags(self, tags, ranker):
        
        tag_scores=[]
        for t in tags:
            sub_vecs={}
            for x in self.sub_docs.keys():
                sub_vec = ranker.score(t, self.sub_docs[x])
                sub_vecs[x]=sub_vec
            tag_scores.append(sub_vecs)
        
        self.subdoc_tag_scores = tag_scores
    
    def rank_titles(self, name, ranker):
        name = re.sub('(,|\.|Inc| )',"",str(name))
        titles = self.query_results['title'].tolist()
        self.title_scores = ranker.score(name, titles)
        
    def rank_ticker(self, ticker, ranker):
        
        #Takes a ticker object and runs all of the rankers above
        
        name = ticker.name
        sym = ticker.ticker
        tags = ticker.tags
        
        self.rank_docs(name,ranker)
        self.rank_doc_tags(tags, ranker)
        self.rank_subdocs(name,ranker)
        self.rank_subdocs_tags(tags,ranker)
        self.rank_titles(name,ranker)
        
    #******************************************************************************
    #----------------------------Pruning Relevant Set------------------------------
    #******************************************************************************
    
    def prune_subdocs(self, cutoff=0.4):
        subbed_data = self.sub_docs
        sub_scores = self.subdoc_scores
        for x in self.sub_docs.keys():

            subbed_data[x] = [xv if c else None for c, xv in zip(sub_scores[x]>cutoff, subbed_data[x])]
            subbed_data[x] = [y for y in subbed_data[x] if y!=None]
            sub_scores[x] = [y for y in sub_scores[x] if y>cutoff]
        
        self.relevant_set = {k: v for k, v in subbed_data.items() if len(v) > 0}
        self.relevant_scores={k: v for k, v in sub_scores.items() if len(v) > 0}
    
    #******************************************************************************
    #-------------------------------------PLSA (from MP3)--------------------------
    #******************************************************************************
    
    def build_term_doc_matrix(self):
        
        m = []
        line = []
        for x in self.documents:
            doc = list(x.split())
            for itm in self.vocabulary:
                line.append(x.count(itm))
            m.append(line)
            line = []
        self.term_doc_matrix = np.array(m)
        
    def initialize_prob(self, number_of_topics):

        self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random_sample((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

            
    def E_step(self):
        
        for x in range(0,self.term_doc_matrix.shape[0]):  #loop through documents
            e = self.document_topic_prob[x].reshape(-1,1)*self.topic_word_prob
            self.topic_prob[x] = normalize(e)
           

    def M_step(self, number_of_topics):
        
        pz = []
        for x in range(0, self.term_doc_matrix.shape[0]):         
            m = self.topic_prob[x]*self.term_doc_matrix[x].reshape(1,-1)
            self.document_topic_prob[x] = np.sum(m,axis=1)
            pz.append(m)

        #update
        
        pz = np.array(pz)
        self.topic_word_prob = np.sum(pz,axis=0)
        
        self.document_topic_prob = normalize(self.document_topic_prob)
        self.topic_word_prob = normalize(self.topic_word_prob)
 

    def calculate_likelihood(self, number_of_topics):

        l = np.log(np.prod(np.power(np.dot(self.document_topic_prob,self.topic_word_prob),self.term_doc_matrix),axis=1))
        l = l[np.argmax(l)]
        self.likelihoods.append(l)
        

    def plsa(self, number_of_topics, max_iter, epsilon):

        self.build_term_doc_matrix()
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)
        self.initialize_prob(number_of_topics)
        current_likelihood = 0.0

        for iteration in range(max_iter):
            self.E_step()
            self.M_step(number_of_topics)
            
            l = self.calculate_likelihood(number_of_topics)
            
            if current_likelihood==0 or current_likelihood==None or l>current_likelihood:
                current_likelihood = l
            else:
                break
