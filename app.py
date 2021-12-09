# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import os
import numpy as np
import pandas as pd
from util.data_manager import data_manager
import webbrowser
from threading import Timer
import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output

port=8050

app = dash.Dash(__name__,)

datadir = os.path.join(os.getcwd(),"_data")
manager = data_manager(datadir=datadir)

tickerlist = manager.get_ticker_list()

app.layout = html.Div([
    html.Div([
        html.H1(children = 'Financial News Sentiment Analysis', 
                style = {'textAlign':'center','marginTop':40,'marginBottom':40}),
        

        dcc.Dropdown(
            id='ticker-dropdown',
            options = [{'label':t, 'value':t} for t in tickerlist],
            value='Select Ticker'),

        dcc.Checklist(
            id='graph-display',
            options=[
                {'label': x, 'value': x} for x in ['scores', 'relevance', 'sentiments']
            ],
            value='scores',
            labelStyle={'display':'inline-block'}
        ),
        
        dcc.Graph(id = 'stock_chart')
         
    ], style={'float':'left','display':'inline-block'})         
])

@app.callback(
    Output('stock_chart','figure'),
    Input('ticker-dropdown', 'value'),
    Input('graph-display','value')
)

def update_graph(ticker, display):
    
    price_df = manager.get_pricedf(ticker=ticker)
#     price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'])
#     price_df.loc[:,'date'] = pd.to_datetime(price_df.loc[:,'date'].dt.strftime('%Y/%m/%d'))
    price_df = price_df.sort_values(by='pub_date')
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,vertical_spacing=0.02, subplot_titles=(ticker,'Sentiment_Score', 'Used News Volume'), 
               row_width=[0.4, 0.6, 1.5], specs=[[{"secondary_y": True}],[{"secondary_y": True}],[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=price_df['pub_date'],
                    open=price_df['open'], high=price_df['high'],
                    low=price_df['low'], close=price_df['close'], name="Price"),
                   secondary_y=True, row=1,col=1)

    
    fig.add_trace(go.Bar(x=price_df['pub_date'], y=price_df['volume'],opacity=0.2, name='Volume'),secondary_y=False)

    if 'scores' in display:
        fig.add_trace(go.Line(x=price_df['pub_date'], y=price_df['scores'], showlegend=True, name='scores', line_color='purple'), row=2, col=1)
    if 'sentiments' in display:
        fig.add_trace(go.Line(x=price_df['pub_date'], y=price_df['sentiments'], showlegend=True, name='sentiments', line_color='lime'), row=2, col=1)
    if 'relevance' in display:
        fig.add_trace(go.Line(x=price_df['pub_date'], y=price_df['relevance'], showlegend=True, name='relevance', line_color='orange'), row=2, col=1)
    
    fig.add_hline(y=0, opacity=0.5, line_width=1, row=2, col=1)
    fig.add_hline(y=1, opacity=0.5, line_dash='dash',line_color='green', row=2, col=1)
    fig.add_hline(y=-1, opacity=0.5, line_dash='dash',line_color='red',row=2,col=1)

    fig.add_trace(go.Bar(x=price_df['pub_date'],y=price_df['Doc_Volume'],opacity=0.2, name='Used News Volume'),row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, width=1000, height=750)
    fig.layout.yaxis2.showgrid=False
    
    return fig


if __name__ == '__main__':
    #Timer(1, open_browser).start();
    webbrowser.open_new("http://localhost:{}".format(port))
    app.run_server(use_reloader=False)
    