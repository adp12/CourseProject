""" 
Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)

Tony: Adjusted for added flexibility
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, norm=None, smooth_idf=True, stopwords=None, b=0.75, k1=1.6, W=[], idf_constant=False, sublinear_tf=False, vocabulary=None):

        self.vectorizer = TfidfVectorizer(norm=norm, smooth_idf=smooth_idf, stop_words=stopwords, sublinear_tf=sublinear_tf, vocabulary=vocabulary)
        self.count_vec = CountVectorizer(stop_words=stopwords, vocabulary=vocabulary)
        self.b = b
        self.W = W
        self.k1 = k1

        self.idf_constant=idf_constant

    def fit(self, X):
        """ Fit to documents """
        self.vectorizer.fit(X)
        self.count_vec.fit(X)
        self.doc_l = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.doc_l.sum(1).mean()

        
    def score(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        if self.idf_constant==True:
            idf = self.vectorizer._tfidf.idf_[None, q.indices]
        else:
            idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.

        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1

    def score_expanded(self, Q, X, weighted_tf=False):
        
        '''
        paper source info:
        2008 Robust and WSD tasks - Perez-Aguera, Zaragoza
        See section 3
        Exploiting morphological query stucture using Genetic Optimisation - Zaragoza, Araujo, Perez-Aguera
        See pages 4-5

        Query Expansion formula:

                                         tf(ci,d) 
        score(d, qc) = SUM(----------------------------------------- * Eidf(ci, d) ) 
                            k1 * (1 - b + b *(|d|/avgdl)) + tf(ci,d)

        tf(ci, d) = sum(tf(t,d))

        Eidf(c,d) = (1 / sum( w(t) tf(d,t) ) ) * sum( w(t) * tf(d,t) * idf(t) )

        for term, weight (t, w) in query clause(c)

        '''

        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = self.count_vec.transform(X)
        dl = X.sum(1).A1
        epsilon = 0.0001
        #Q should be a list of query weight pairs
        #Loop through the queries in Q
        #each q in Q should be a [query, weight]
        self.tfc=[]
        self.Eidf=[]
        for q in Q:
            qe, = self.count_vec.transform([q[0]])
            tf = X.tocsc()[:,qe.indices]
            wtf = X.tocsc()[:,qe.indices] * q[1]
            if weighted_tf:
                self.tfc.append(wtf)
            else:
                self.tfc.append(tf)
            idf = self.vectorizer._tfidf.idf_[None, qe.indices] - 1.
            eidf = ( float( 1/ (wtf.sum()+epsilon) ) * wtf.multiply(np.broadcast_to(idf, wtf.shape))).sum()
            
            self.Eidf.append(eidf)

        scr=[]
        for x in range(0, len(Q)):
            denom = self.tfc[x] + (k1 * (1 - b + b * (dl/avdl)))[:,None]
            numer = self.tfc[x].multiply(np.broadcast_to(self.Eidf[x], self.tfc[x].shape))
            scr.append((numer/denom).sum(1).A1)

        output_score = np.sum(scr, axis=0)

        return output_score

