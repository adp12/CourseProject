import requests
import re
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil import parser as dparser

class Ticker(object):
    
    def __init__(self, config, t, source='alpha'):
        #Get general information about company from stock ticker symbol via api
        #name, ticker, industry, sector, tags
        
        t = t.upper()
        self.config = config

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

        name = re.sub('(Inc|Company|Corporation|Corp|Co|Incorporated|LLC|Group|Limited|,|\.)',"",str(self.name), flags=re.I)
        self.name_adj = name

        

    def get_price(self, days=100, d_start=None, source='alpha'):
        #get price information from yahoo finance api
        #return a pandas dataframe for either a standard line chart[date,price] or a candlestick chart[date,open,high,low,close,volume]

        #determining how much data to ask for
        if d_start==None:
            if days<=60:
                asize='compact'
                drange='3mo'
            elif days<=100:
                asize='compact'
                drange='6mo'
            else:
                asize='full'
                drange='1y'
        else:
            d_start = dparser.parse(d_start)
            alpha_datecut = datetime.now()-relativedelta(days=150)
            if d_start<alpha_datecut:
                #all datapoints
                asize = 'full'
                drange = '1yr'
            else:
                #last 100 datapoints
                asize = 'compact'
                drange = '6mo'

        if source == 'alpha':
            url = ('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY'
                    '&symbol='+self.ticker+
                    '&outputsize='+asize+
                    '&apikey='+self.config.alpha)
                
            response = requests.get(url, timeout=10)
            if response.status_code==200:
                response = response.json()
                date=[]
                volume=[]
                high=[]
                low=[]
                opn=[]
                cls=[]
                for i in response['Time Series (Daily)']:
                    date.append(i)
                    opn.append(response['Time Series (Daily)'][i]['1. open'])
                    high.append(response['Time Series (Daily)'][i]['2. high'])
                    low.append(response['Time Series (Daily)'][i]['3. low'])
                    cls.append(response['Time Series (Daily)'][i]['4. close'])
                    volume.append(response['Time Series (Daily)'][i]['5. volume'])

                price_df = pd.DataFrame()
                price_df['date']=date
                price_df['high']=high
                price_df['low']=low
                price_df['open']=opn
                price_df['close']=cls
                price_df['volume']=volume
                price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'])
                price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'].dt.strftime('%Y/%m/%d'))

                return price_df

        elif source == 'yahoo':
            drange='6mo'
            interval='1d'
            chart='candle'

            yheaders = {'x-api-key':self.config.yahoo}
            if chart=='line':
                
                url = ('https://yfapi.net/v8/finance/spark?'
                        'interval='+interval+
                        '&range='+drange+
                        '&symbols='+self.ticker)
                response = requests.request("GET",url,headers=yheaders)
                if response.status_code==200:
                    response = response.json()

                    date = response[self.ticker]['timestamp']
                    price = response[self.ticker]['close']
                    price_df = pd.DataFrame()
                    price_df['date']=date
                    price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'], unit='s')
                    price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'].dt.strftime('%Y/%m/%d'))
                    price_df['price']=price
                    price_df = price_df.set_index('date')

                    return price_df

            elif chart=='candle':
                #send request to yahoo finance for daily prices
                url=('https://yfapi.net/v8/finance/chart/'+self.ticker+"?"+
                    'comparisons='+self.ticker+
                    '&range='+drange+
                    '&region=US'
                    '&interval='+interval+
                    '&lang=en')
                response = requests.request("GET",url,headers=yheaders)

                if response.status_code==200:
                    response = response.json()            
                    date = response['chart']['result'][0]['timestamp']
                    volume = response['chart']['result'][0]['indicators']['quote'][0]['volume']
                    high = response['chart']['result'][0]['indicators']['quote'][0]['high']
                    low = response['chart']['result'][0]['indicators']['quote'][0]['low']
                    opn = response['chart']['result'][0]['indicators']['quote'][0]['open']
                    cls = response['chart']['result'][0]['indicators']['quote'][0]['close']
                    #put data into pandas df
                    price_df = pd.DataFrame()
                    price_df['date']=date
                    price_df['high']=high
                    price_df['low']=low
                    price_df['open']=opn
                    price_df['close']=cls
                    price_df['volume']=volume
                    price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'], unit='s')
                    price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'].dt.strftime('%Y/%m/%d'))

                    return price_df

            
        return response
            

