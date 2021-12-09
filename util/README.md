# Setting Up API Keys

## Overview
In order to protect sensitive information such as the API keys used in config.py, we have masked them by storing the actual keys in a local file that is ignored by git. We recommend you do the same.

## keys.py
Create a local file named 'keys.py' and populate it as shown below, replacing myAPIKey with your own generated API keys. Subsequently, config.py will call the keys.py class to obtain the API keys. A list of these APIs and information about them can be found detailed in the root README.md. Do NOT commit keys.py to your GitHub source code repository.

```
class keys(object):
    def __init__(self):
        #News APIs
        self.newsapi = 'myAPIKey'
        self.guardian = 'myAPIKey'
        self.currents = 'myAPIKey'
        self.usearch_host = 'myAPIKey'
        self.usearch_key = 'myAPIKey'
        #Finance Specific APIs
        self.polygon = 'myAPIKey'
        self.yahoo = 'myAPIKey'
        self.alpha = 'myAPIKey'
```


![alt](../img/meme1.jpg)

