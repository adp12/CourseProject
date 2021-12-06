import numpy as np
import pandas as pd
import os

class data_manager(object):

    def __init__(self, datadir):
        self.datadir = datadir
        self.ticker_list = None

    def get_ticker_list(self):
        for _, dirname, _ in os.walk(self.datadir):
            tickerlist = dirname
            break
        self.ticker_list = tickerlist
        return tickerlist

    def get_data_info(self, ticker):
        #get a dictionary of info on the current stored data
        if self.ticker_list is None or len(self.ticker_list)==0:
            self.get_ticker_list()

        if ticker in self.ticker_list:
            data = {}
            #if there's stuff in the ticker folder
            tdir = os.path.join(self.datadir,ticker)
            if len(os.listdir(tdir)) > 0:
                #check if full_df is there
                f = os.path.join(tdir,'full_df.csv')
                if os.path.exists(f):
                    full_df = pd.read_csv(f, index_col=False)
                    #ensure datetime
                    full_df['pub_date'] = pd.to_datetime(full_df.pub_date)
                    
                    mindate = np.min(full_df['pub_date'])
                    maxdate = np.max(full_df['pub_date'])
                    rows = len(full_df)
                    data['full_df'] = {'mindate':mindate,'maxdate':maxdate,'rows':rows}

                #check if price_df is there
                f = os.path.join(tdir,'price_df.csv')
                if os.path.exists(f):
                    price_df = pd.read_csv(f, index_col=False)
                    price_df['pub_date'] = pd.to_datetime(price_df.pub_date)
                    mindate = np.min(price_df['pub_date'])
                    maxdate = np.max(price_df['pub_date'])
                    rows = len(price_df)
                    data['price_df'] = {'mindate':mindate,'maxdate':maxdate,'rows':rows}

                if len(data)>0:
                    return data
        return 0

    def store_data(self, ticker, full_df=None, price_df=None):
        
        
        self.get_ticker_list()
        
        #check if there is a folder for this ticker, if not create one
        #assign variable tdir as the ticker's directory to save out files
        if ticker not in self.ticker_list:
            newdir = os.path.join(self.datadir,ticker)
            os.makedirs(newdir)
            tdir = newdir
        else:
            tdir = os.path.join(self.datadir,ticker)

        #if there's stuff in the ticker folder, add to it
        if len(os.listdir(tdir)) > 0:
            if full_df is not None:
                f = os.path.join(tdir, "full_df.csv")
                if os.path.exists(f):
                    stored_df = pd.read_csv(f)
                    stored_df['pub_date'] = pd.to_datetime(stored_df.pub_date)
                    #for now, append new full_df to stored_df and remove duplicate dates
                    #then resave
                    df = stored_df.append(full_df)
                    
                    df = df.drop_duplicates(subset=['pub_date','url'],keep='last')
                    
                    df.to_csv(f, index=True, index_label='index')
                else:
                    full_df.to_csv(f, index=True, index_label='index')
            
            if price_df is not None:
                f = os.path.join(tdir, "price_df.csv")
                if os.path.exists(f):
                    stored_df = pd.read_csv(f)
                    stored_df['pub_date'] = pd.to_datetime(stored_df.pub_date)
                    #same as above
                    df = stored_df.append(price_df)
                    df = df.drop_duplicates(subset=['pub_date'],keep='last')
                    
                    df.to_csv(f, index=True, index_label='index')
                else:
                    price_df.to_csv(f, index=True, index_label='index')

        #if not store them
        else:
            if full_df is not None:
                f = os.path.join(tdir, "full_df.csv")
                full_df.to_csv(f, index=True, index_label='index')
            
            if price_df is not None:
                f = os.path.join(tdir, "price_df.csv")
                price_df.to_csv(f, index=True, index_label='index')