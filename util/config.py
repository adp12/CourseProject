from util.keys import keys

keys=keys()

class config(object):
    def __init__(self):

        #News APIs
        self.newsapi = keys.newsapi
        self.currents = keys.currents
        self.usearch_host = keys.usearch_host
        self.usearch_key = keys.usearch_key
        #Finance Specific APIs
        self.polygon = keys.polygon
        self.yahoo = keys.yahoo
        self.alpha = keys.alpha