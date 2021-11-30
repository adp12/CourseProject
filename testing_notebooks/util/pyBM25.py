""" 
Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)

Has been modified for additional flexibility
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, norm=None, smooth_idf=True, stopwords=None, b=0.75, k1=1.6, idf_constant=False,  sublinear_tf=False, vocabulary=None):
        self.vectorizer = TfidfVectorizer(norm=norm, smooth_idf=smooth_idf, stop_words=stopwords, sublinear_tf=sublinear_tf, vocabulary=vocabulary)
        self.b = b
        self.k1 = k1
        self.idf_constant=idf_constant

    def fit(self, X):
        # Fit IDF to documents X
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def score(self, q, X):
        # Calculate BM25 between query q and documents X
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be converted to
        # idf(t) = log [ n / df(t) ] - 1
        if self.idf_constant==True:
            idf = self.vectorizer._tfidf.idf_[None, q.indices]
        else:
            idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.

        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1

