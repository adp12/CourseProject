# CS_410 CourseProject

Team AHR
***
Members

●	Anthony Petrotte (adp12@illinois.edu)

●	Hrishikesh Deshmukh (hcd3@illinois.edu)

●	Rahul Jonnalagadda (rjonna2@illinois.edu)

***

# Description


The goal of our project is firstly, to analyze the financial news cycle in different time intervals to create a time interval sentiment metric on particular global securities. Secondly, we would compile the intermediate sentiment results during the metric calculation into a time series dataset that can be compared to price movement in the underlying security. This dataset would potentially have many uses, including the possibility to aid in identifying securities driven by trades focused around the new cycle. Another interesting use could be to identifying securities that are more prone to volatility from volume in individual (and potentially more naïve) investors trading in a more impulsive/emotional manner due to recently read information.

By collecting a corpus of recent news focused a specific company/security and trimming it down into subsequently smaller relevant sub-documents, we aim to create a time-series sentiment calculation that can be compared to the price of the company/security. Although the collected data may have many potential uses, we have opted to use the data graphically as a time series. If you are familiar with the concept of a stock price indicator/oscillator, the resulting dataset can be used in a similar manner. This would be a useful tool or addition to the task of stock screening, and could be implemented as an addition to a computational trading strategy.


***
# General Use
The general methodology to use this code base is to 1.) Build Data and 2.) Visualize the data you have compiled

### Building Data Set
The `data_creation` notebook serves as a fully functional example of how the code base is used to pipeline local datasets to `_data`

### Visualizing
We have provided a plotly dash app to interactively graph the datasets you build!

There are two ways to run the dash app in the repo:

* `python app.py`: from the terminal, which will automatically open your browser to the correct local host address
* `dash_app.ipynb`: a jupyter notebook version of `app.py`. When the cell for the dash is run, it will give you a hyper link to the local host address with your dashboard


# Setup

*This section highlights what is needed to run locally and helpful links to get you started*
***

### Dependencies
*This project has been implemented with python and uses the following open libraries (listed with their use)*

* Data Processing
  * numpy
  * pandas
  * sklearn
  * scipy
  * re
  * os
  * math
* Online Data Retrieval
  * BeautifulSoup
  * requests
  * urllib 
* Date/time formating
  * datetime
  * dateutil
  * time 
  * timeit
* DNN for NLP (sentiment analysis)
  * torch
  * transformers
* Graphing/Utilizing
  * plotly
  * dash

*each of these libraries can be installed via pip or by most typical methods used for installing python libraries*

***
### Initial Setup
  

#### Needed:
* API to get news data
* API to get ticker information

#### Recommended APIs to get started

##### API to get news data
* [Polygon.io](https://polygon.io/dashboard/signup)
  * Free for news (pricey for market data)
  * Best option for quickly getting started compiling news
  * Very large page size limit (1000)
  * Not good for company info, especially outside of the US

##### API to get ticker information
* [AlphaVantage](https://www.alphavantage.co/support/#api-key)
  * Free
  * Easy
 
### Supported APIs
* [NewsApi](https://newsapi.org/register) (formerly google, now NewsAPI.org)
  * Free Developer Version
  * Only past 1 month's news
  * Daily call limit: 100
  * Page size limit: 100
* [Currents API](https://currentsapi.services/en/register)
  * Free
  * Daily call limit: 600
  * Page size limit: 200
* [Usearch](https://rapidapi.com/contextualwebsearch/api/web-search) (formerly contextual_web, now Web Search on RapidAPI)
  * Need Rapid API account
  * Free-mium
  * Daily call limit: 100
  * Page size limit: 50
  * Rate limit: 1 per second
* [Yahoo Finance](https://www.yahoofinanceapi.com)
  * Good for quick price info
  * Was previously good for getting company info before it stoppped working all of a sudden


***
#### Important Note on API keys
**The API keys need put into the config.py file under their respective variables.**
**Please See details in the [util](./util/) folder for setting up the config**
***



# Method
*This section goes into detail about the method used to compile web results and extract sentiments and serves as a how to guide for using this code base*

*In addtion there are two diagrams of both the entire model as well as a more focused diagram of the document to relevant sub-document process in the [img](./img/) subdirectory*
***

### Setup
* Pull in api keys from config making a config variable `config=config()`
  * This variable will need to be passed to any class that needs to call an api 
* create a `Ticker()` object: passing in config
* Initiate a `Corpus()` object
### Getting web results and initiating Corpus
#### Using web_query:
* Initiate a web_query object
  * `query_all()`: Runs through the available apis and makes threaded calls to collect result urls
  * `compile_results()`: combines the results of the api responses and deletes duplicate urls
  * `scrape_results()`: scrapes the list of urls and compiles the raw website text to be given to the Corpus
  * `get_results()`: returns the stored dataframe of urls and text
#### Setting up the Corpus:
* Use the `set_results()` function to store the results from the web_query in the corpus for processing (needed for dataset building)
* Use `set_corpus()` with the web_query documents and urls
### Setting up Ranker and Initial Queries
* Initiate the ranking function (in our case, custom built BM25 object class from util.pyRanker)
* Fit the ranker to the corpus documents
* Build the queries used by the ranking function using `build_queries()` and passing in the ticker object
### Initial Ranking
* Rank corpus documents for relevance with `rank_docs()` passing in the ranker
* Prune the documents with `prune_docs()`. This is pre-set to only prune 0 ranked documents on the primary query using a standard BM25 score
  * Note: these documents are not removed, only indexed in the pruned_docs for the sub-division proecess
### Sub-Dividing
* Create dictionary of sub-docs from the original documents by calling `sub_divide()` and passing in the Transformer's tokenizer(if needed)
  * Note: The tokenizer is needed to ensure that the length of the subdocs created will not exceed the maximum token size used by the Transformer. The Corpus sets the standard distil-BERT tokenizer on initiation.
* Rank the newly created subdocs using `rank_subdocs()` and pass in the ranker
* Prune the subdocs with `prune_subdocs()`. This is pre-set similarly to `prune_docs()` and prunes 0 ranked subdocs using the prime query and standard BM25 score
  * Note: Like the `prune_docs()` function, this does not remove any subdocs, but creates an index of the ones to keep from the subdoc dictionary
### Relevant Set
* after sub-dividing and pruning out all the trash, make the relevant set by calling `make_relevant()`
* Rank the relevant set with `rank_relevant()`.
  * This ranking function is pre-set to run a BM25 with Structured Query Expansion. The expanded query and its weights set can be adjusted when initiating `build_queries()`
* If needed/wanted, you can further prune the relevant set using `prune_relevant()`
  * Note: unlike the other two pruning functions, this directly adjusts the relevant_set and relevant_scores stored by the corpus
### Sentiments
* `get_sentiments()`: Once a relevant set is established
  * Using Transformer's classifier will run each relevant subdoc through the classifier to produce a sentiment score
### Dataset Initiation
* `data_preprocess()` will setup the needed dictionaries for creating pandas dataframes
  * This is also where the scores are calculated, which are simply the sentiment weighted by the relevance
* build the two dataframes with `build_fulldf()` and `build_pricedf()`
### Dataset for graphing
* initiate a data_manager object from util.data_manager.
  * the `data_manager()` needs the location of the '_data' directory on initiation
* tell the data_manager to put the data in a retrievable place with `data_manager.store_date()` and pass in the ticker symbol and dataframes
### Graphing
* `get_ticker_list()` calls the datamanger to return the available list of stored tickers 
* `get_pricedf()` will tell the datamanger to return the stored price_df.csv as a pandas dataframe. This should have everything needed to graph
* `get_fulldf()` will return the full_df.csv. This is all of the data needed for recalculation, or recompiling the relevant documents as it holds urls


***
**Note:**
All used source sites are in the references sub-directory
In addition, all papers that were foundational in the creation of the code base are in the references sub-directory in pdf format (for those who want a little light reading)
