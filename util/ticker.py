import requests
import re


class Ticker(object):
    
    def __init__(self,config, t, source='best'):
        #Get general information about company from stock ticker symbol via api
        #name, ticker, industry, sector, tags
        
        t = t.upper()
        if source=='best':
            #Try to get ticker information from Polygon.io
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
                #Try to get ticker information from yahoofinance
                print("Bad reponse from Polygon.io: Status_code:"+str(response.status_code))
                print("Trying Yahoo Api")

                yheaders = {'x-api-key':config.yahoo}
                url = ('https://yfapi.net/v11/finance/quoteSummary/'+t+'?lang=en&region=US&modules=assetProfile')
                url2 = ('https://yfapi.net/v6/finance/quote?symbols='+t)
                response1 = requests.request("GET",url,headers=yheaders).json()
                response2 = requests.request("GET",url2,headers=yheaders).json()
                
                self.sector = response1['quoteSummary'].get('result')[0].get('assetProfile').get('sector')
                self.industry = response1['quoteSummary'].get('result')[0].get('assetProfile').get('industry')
                self.name = response2['quoteResponse'].get('result')[0].get('longName')
                self.ticker = response2['quoteResponse'].get('result')[0].get('symbol')
                t1 = re.split('\n|\.|&|,|and',self.sector)
                t2 = re.split('\n|\.|&|,|and',self.industry)
                tags=[]
                for i in t1:
                    tags.append(i.strip())
                for i in t2:
                    tags.append(i.strip())
                self.tags=tags
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

        elif source=='yahoo':
            #Try to get ticker information from yahoofinance when specified
            yheaders = {'x-api-key':config.yahoo}
            url = ('https://yfapi.net/v11/finance/quoteSummary/'+t+'?lang=en&region=US&modules=assetProfile')
            url2 = ('https://yfapi.net/v6/finance/quote?symbols='+t)
            response1 = requests.request("GET",url,headers=yheaders)

            if response1.status_code==200:
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
            else:
                print("Bad response from yahoo finance, affecting sector/industry assignments and subsequent tagging")

            response2 = requests.request("GET",url2,headers=yheaders)

            if response2.status_code==200:
                response2 = response2.json()
                self.name = response2['quoteResponse'].get('result')[0].get('longName')
                self.ticker = response2['quoteResponse'].get('result')[0].get('symbol')
            else:
                print("Bad response from yahoo finance, affecting name/ticker assignment")
    
    def get_price(self, config, drange='1mo', interval='1d'):
        
        yheaders = {'x-api-key':config.yahoo}
        url = ('https://yfapi.net/v8/finance/spark?'
                'interval='+interval+
                '&range='+drange+
                '&symbols='+self.ticker)
        response = requests.request("GET",url,headers=yheaders)
        if response.status_code==200:
                response = response.json()
            
        return response
            

