from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dateutil import parser as dparser

import ipywidgets as widgets
import pandas as pd
import re
import requests

from time import time
import timeit
import urllib.parse

class web_query(object):
    '''
    **********************************************************************************
    *********Large wrapper class for querying apis to accumulate news data************
    **********************************************************************************

    Method:

    web_query(config_file)
    *Instantiate web_query object by passing in a config file containing the api keys
    **config file currently in util directory

    query_all()
    *Can query specific apis with more specifications, or use query_all() to run
    through all the apis with page related default parameter (recommended)

    compile_results()
    *combines the results of the various queries into a singular pandas dataframe
    *stored in self.results
    *also removes any duplicate url or title instance
    **dateframe columns('title', 'url', 'pub_date')
    
    scrape_results(max_docs)
    *scrapes the text from the urls accumulated in the query results up to the specified
    number of max_docs (default 50)
    *scraped text stored in self.documents
    ***this can take some time

    **********************************************************************************
    '''


    def __init__(self, config):
        self.config = config
        self.frames = []
        self.results=None
        self.documents=[]
        self.number_of_documents=None
        self.failed=[]

        self.timer=[]
    
    def get_results(self):
        return self.results
    
    def get_urllist(self):
        return self.results['url'].tolist()
    
    #added multi-threading to improve scrape speed 
    #Estimated performance increase over iterative: 85%

    def thread_scrape(self, pgres, soup_parser, url):
        len_min=200
        headers = {"User-Agent":"Mozilla/5.0"}
        start = timeit.default_timer()
        try:
            response = requests.get(url=url, headers=headers, timeout=5)
            #if response ok and no errors
            if response.status_code==200:
                soup = BeautifulSoup(response.content, soup_parser)
                d = soup.get_text()
                if len(d)>len_min:
                    self.documents.append(d)
            #else track failures
            else:
                self.failed.append(url)
        except:
            self.failed.append(url)
        finally:
            stop = timeit.default_timer()
            pgres.value+=1
        self.timer.append((stop-start))


    def scrape_results(self, threaded=True, max_docs=50, workforce=20, soup_parser='html.parser'):
        
        url_list = self.results['url'].tolist()
        url_list = url_list[0:max_docs]
        len_min=200
        times = []
        failed = []

        pgres = widgets.IntProgress(value=0,min=0,max=len(url_list), step=1)
        display(pgres)
        
        s = requests.Session()
        headers = {"User-Agent":"Mozilla/5.0"}
        s.headers.update(headers)

        #Need multi-threading to improve speed
        if threaded:
            with ThreadPoolExecutor(max_workers=workforce) as executor:
                [executor.submit(self.thread_scrape, pgres=pgres, soup_parser=soup_parser, url=x) for x in url_list]
        else:
            for i in range(0,len(url_list)):
                start = timeit.default_timer()
                try:
                    response = s.get(url=url_list[i])
                    #if response ok and no errors
                    if response.status_code==200:
                        soup = BeautifulSoup(response.content, soup_parser)
                        d = soup.get_text()
                        if len(d)>len_min:
                            self.documents.append(d)
                    #else track failures
                    else:
                        self.failed.append(i)
                except:
                    self.failed.append(i)
                finally:
                    pgres.value+=1
                    pgres.description=str(i+1)+":"+str(len(url_list))
                
                stop = timeit.default_timer()
                times.append((stop-start))

            self.timer = times        
        self.number_of_documents=len(self.documents)
        #remove failed url responses from dataset
        self.results = self.results.take(list(set(range(self.results.shape[0]))-set(self.failed)))

    def compile_results(self):
        #return results as pandas dataframe
        results = pd.concat(self.frames)
        results = results.drop_duplicates(subset=['url'])
        results = results.drop_duplicates(subset=['title'])
        self.results=results
    

    def query_all(self, query, ticker, d_start='Now', d_end='Now', threaded=True):
        
        if threaded:
            apis = ['usearch', 'poly', 'currents', 'newsapi']
            with ThreadPoolExecutor(max_workers=len(apis)) as executor:
                    [executor.submit(self.query_api, query=query, ticker=ticker, d_start=d_start, d_end=d_end, api=x) for x in apis]
        else:
            # #Running query through Usearch API
            self.query_Usearch(query=query, d_start=d_start)
            
            # #Running query through Currents API
            self.query_currents(query=query, d_start=d_start)
            
            # #Running query through Polygon.io
            self.query_polygon(ticker=ticker, d_start=d_start)
            
            # #Running query through (google)NewsAPI
            self.query_newsapi(query=query, d_start=d_start)
            
        
    def query_api(self, query, api, ticker, d_start='Now', d_end='Now', page=1):
        #Used for multi-processing the api requests to improve time
        #Estimated performance increase over iterative: 40%

        if api=='usearch':
            self.query_Usearch(query=query, d_start=d_start, d_end=d_end)

        elif api=='poly':
            self.query_polygon(ticker=ticker, d_start=d_start, d_end=d_end)

        elif api=='currents':
            self.query_currents(query=query, d_start=d_start, d_end=d_end)

        elif api=='newsapi':
            self.query_newsapi(query=query, d_start=d_start, d_end=d_end)

    def query_Usearch(self, query, d_start='Now', d_end='Now', page=1, pageSize=50):
        query = urllib.parse.quote_plus(query)
        #Valid format : Date format should be YYYY-MM-ddTHH:mm:ss.ss±hh:mm
        if d_end=='Now':
            d_end=datetime.now()
        else:
            d_end = dparser.parse(d_end)
        d_end_str = d_end.strftime("%Y-%m-%d")

        # Same thing but for start date
        if d_start=='Now':
            d_start=datetime.now()
        else:
            d_start = dparser.parse(d_start)
        d_start_str = d_start.strftime("%Y-%m-%d")

        url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
        querystring = {"q":query,
                       "pageNumber":"1",
                       "pageSize":"150",
                       "autoCorrect":"false",
                       "fromPublishedDate":d_start_str,
                       "toPublishedDate":d_end_str}

        headers = {
            'x-rapidapi-host': str(self.config.usearch_host),
            'x-rapidapi-key': str(self.config.usearch_key)
            }

        response = requests.request("GET", url, headers=headers, params=querystring)

        #Check that reponse is valid
        if response.status_code==200:
            response = response.json()

            #Loop through response to create return Dataframe: columns: Titles, Urls, Publication Dates
            datalist=[]
            for x in response['value']:
                row = {'title':str(x['title']), 'url':x['url'], 'pub_date':x['datePublished']}
                datalist.append(row)
            df = pd.DataFrame.from_dict(datalist)
            self.frames.append(df)
    
    def query_currents(self, query, d_start='Now', d_end='Now', page=1, pageSize=200):
        #6-Month archive

        query = urllib.parse.quote_plus(query)

        #Valid format : Date format should be YYYY-MM-ddTHH:mm:ss.ss±hh:mm
        if d_end=='Now':
            d_end=datetime.now()
        else:
            d_end = dparser.parse(d_end)
        d_end_str = d_end.strftime("%Y-%m-%d")

        if d_start=='Now':
            d_start=datetime.now()
        else:
            d_start = dparser.parse(d_start)
        d_start_str = d_start.strftime("%Y-%m-%d")


        url = ('https://api.currentsapi.services/v1/search?'
               '&start_date='+d_start_str+
               '&end_date='+d_end_str+
               '&keywords='+query+
               '&language=en'
               '&country=us'
               '&page_number='+str(page)+
               '&page_size='+str(pageSize)+
               '&apiKey='+str(self.config.currents))

        #Check that reponse is valid
        response = requests.get(url)
        if response.status_code==200:
            response = response.json()

            #Loop through response to create return Dataframe: columns: Titles, Urls, Publication Dates
            datalist=[]
            for x in response['news']:
                row = {'title':str(x['title']), 'url':x['url'], 'pub_date':x['published']}
                datalist.append(row)
            df = pd.DataFrame.from_dict(datalist)
            self.frames.append(df)
    
    def query_polygon(self, ticker, d_start='Now', d_end='Now', pageSize=200):
        ticker = ticker.upper()

        #Valid format : Date format should be YYYY-MM-ddTHH:mm:ss.ss±hh:mm
        if d_end=='Now':
            d_end=datetime.now()
        else:
            d_end = dparser.parse(d_end)
        d_end_str = d_end.strftime("%Y-%m-%d")

        # Same thing but for start date
        if d_start=='Now':
            d_start=datetime.now()
        else:
            d_start = dparser.parse(d_start)
        d_start_str = d_start.strftime("%Y-%m-%d")


        url = ('https://api.polygon.io/v2/reference/news?'
          'ticker='+ticker+
          '&published_utc/gte='+d_start_str+
          '&published_utc/lte='+d_end_str+
          '&limit='+str(pageSize)+
          '&sort=published_utc'
          '&apikey='+str(self.config.polygon))

        #Check that reponse is valid
        response = requests.get(url)
        if response.status_code==200:
            response = response.json()

            #Loop through response to create return Dataframe: columns: Titles, Urls, Publication Dates
            datalist=[]
            for x in response['results']:
                row = {'title':str(x['title']), 'url':x['article_url'], 'pub_date':x['published_utc']}
                datalist.append(row)
            df = pd.DataFrame.from_dict(datalist)
            self.frames.append(df)
    
    def query_newsapi(self, query, d_start='Now', d_end='Now', domains="", exclude="", page=1, pageSize=100):
        #1 Month archive

        query = urllib.parse.quote_plus(query)

        #Valid format : Date format should be YYYY-MM-ddTHH:mm:ss.ss±hh:mm
        if d_end=='Now':
            d_end=datetime.now()
        else:
            d_end = dparser.parse(d_end)
        d_end_str = d_end.strftime("%Y-%m-%d")

        # Same thing but for start date
        if d_start=='Now':
            d_start=datetime.now()
        else:
            d_start = dparser.parse(d_start)
        d_start_str = d_start.strftime("%Y-%m-%d")

        url = ('https://newsapi.org/v2/everything?'
          'q='+query+
          '&domains='+domains+
          '&excludeDomains='+exclude+
          '&from='+d_start_str+
          '&to='+d_end_str+
          '&language=en'
          '&sortBy=publishedAt'
          '&pageSize='+str(pageSize)+
          '&page='+str(page)+  
          '&apikey='+str(self.config.newsapi))

        #Check that reponse is valid
        response = requests.get(url)
        if response.status_code==200:
            response = response.json()

            #Loop through response to create return Dataframe: columns: Titles, Urls, Publication Dates    
            datalist=[]
            for x in response['articles']:
                row = {'title':str(x['title']), 'url':x['url'], 'pub_date':x['publishedAt']}
                datalist.append(row)
            df = pd.DataFrame.from_dict(datalist)
            self.frames.append(df)
