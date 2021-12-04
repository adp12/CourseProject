import requests
import re
import pandas as pd


class Ticker(object):
    
    def __init__(self, config, t, source='alpha'):
        #Get general information about company from stock ticker symbol via api
        #name, ticker, industry, sector, tags
        
        t = t.upper()

        if source=='yahoo':
            #Try to get ticker information from yahoofinance when specified
            yheaders = {'x-api-key':config.yahoo}

            url = ('https://yfapi.net/v11/finance/quoteSummary/'+t+'?lang=en&region=US&modules=assetProfile')
            url2 = ('https://yfapi.net/v6/finance/quote?symbols='+t)
            response1 = requests.request("GET",url,headers=yheaders)

            if response1.status_code==200:
                try:
                    response1 = response1.json()
                    self.sector = response1['quoteSummary'].get('result')[0].get('assetProfile').get('sector')
                    self.industry = response1['quoteSummary'].get('result')[0].get('assetProfile').get('industry')
                    t1 = re.split('\n|\.|&|,|and',self.sector)
                    t2 = re.split('\n|\.|&|,|and',self.industry)
                    tags=[]
                    for i in t1:
                        tags.append(i.strip())
                    for i in t2:
                        tags.append(i.strip())
                    self.tags=tags
                except:
                    print("Bad response from yahoo finance, affecting sector/industry assignments and subsequent tagging")
                    
            else:
                print("Bad response from yahoo finance, affecting sector/industry assignments and subsequent tagging")

            response2 = requests.request("GET",url2,headers=yheaders)

            if response2.status_code==200:
                try:
                    response2 = response2.json()
                    self.name = response2['quoteResponse'].get('result')[0].get('longName')
                    self.ticker = response2['quoteResponse'].get('result')[0].get('symbol')
                except:
                    print("Bad response from yahoo finance, affecting name/ticker assignment")    
            else:
                print("Bad response from yahoo finance, affecting name/ticker assignment")

        elif source=='poly':
            #Try to get ticker information from Polygon.io when specified
            url = ('https://api.polygon.io/v1/meta/symbols/'+t+'/company?apikey='+config.polygon)
            response = requests.get(url)
            
            if response.status_code==200:
                response = response.json()
                self.name = response['name']
                self.tags = response['tags']
                self.industry = response['industry']
                self.sector = response['sector']
                self.ticker = response['symbol']
            else:
                print('Bad response from Polygon.io')

        elif source=='alpha':

            url = ('https://www.alphavantage.co/query?function=OVERVIEW'
                    '&symbol='+t+
                    '&apikey='+config.alpha)
            response = requests.get(url)

            if response.status_code==200:
                response = response.json()
                self.name = response['Name']
                self.industry = response['Industry']
                self.sector = response['Sector']
                self.ticker = response['Symbol']
                
                t1 = re.split('\n|\.|&|,|and',self.sector)
                t2 = re.split('\n|\.|&|,|and',self.industry)
                tags=[]
                for i in t1:
                    tags.append(i.strip())
                for i in t2:
                    tags.append(i.strip())
                self.tags=tags

        name = re.sub('(Inc|Company|Corporation|Corp|Co|Incorporated|LLC|,|\.| )',"",str(self.name), flags=re.I)
        self.name_adj = name

        

    

    def get_price(self, config, drange='1mo', interval='1d', chart='line'):
        #get price information from yahoo finance api
        #return a pandas dataframe for either a standard line chart[date,price] or a candlestick chart[date,open,high,low,close,volume]
        yheaders = {'x-api-key':config.yahoo}
        if chart=='line':
            
            url = ('https://yfapi.net/v8/finance/spark?'
                    'interval='+interval+
                    '&range='+drange+
                    '&symbols='+self.ticker)
            response = requests.request("GET",url,headers=yheaders)
            if response.status_code==200:
                    response = response.json()

        elif chart=='candle':
            url=('https://yfapi.net/v8/finance/chart/'+self.ticker+"?"+
                'comparisons='+self.ticker+
                '&range='+drange+
                '&region=US'
                '&interval='+interval+
                '&lang=en')
            response = requests.request("GET",url,headers=yheaders)
            if response.status_code==200:
                    response = response.json()            

            
        return response
            

