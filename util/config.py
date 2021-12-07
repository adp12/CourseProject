import util.keys as keys

class config(object):
    def __init__(self):
        #keys = keys()
        #News APIs
        self.newsapi = keys.newsapi
        self.guardian = keys.guardian
        self.currents = keys.currents
        self.usearch_host = keys.usearch_host
        self.usearch_key = keys.usearch_key
        #Finance Specific APIs
        self.polygon = keys.polygon
        self.yahoo = keys.yahoo
        self.alpha = keys.alpha