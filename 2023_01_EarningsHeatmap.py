"""
@author: eirik
@date: 2023-05-12

Research doc for "Impact of relative earnings on volatility and returns in equities"
"""


# Module 1: Distribution of ttm earnings

def m1():
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = pd.read_csv('../Database/master__netinc.csv')

    data['Prev TTM Net Income'] = data['TTM Net Income'].shift(4)

    data = data[data['TTM Net Income']>= 0]
    data = data[data['Prev TTM Net Income'] >= 0]

    data['chg'] = data['TTM Net Income']/data['Prev TTM Net Income'] - 1
    data.dropna(how='any', inplace=True)

    # Smooth the data
    data['adj chg'] = data['chg']
    data.loc[data['chg'] >= 0.32, 'adj chg'] = 0.32
    data.loc[data['chg'] <= -0.20, 'adj chg'] = -0.20

    sns.displot(data, x= 'adj chg', kde = True, binwidth = 0.04)
    plt.show()

    print(data.head(50))
    print(data.info())



# Module 2: 

def scrape_mt(link):

    import pandas as pd
    from bs4 import BeautifulSoup
    import requests
    import json
    
    dict_items = {}
    html = requests.get(link, timeout = 10)
    soup = BeautifulSoup(html.text, 'html.parser')
    htmltext = soup.prettify()

    items = htmltext.split("var ")[1:]
    for i in items:
        name = i.split("=")[0].strip()
        try:
            value = i.split("originalData =")[1]
            value = value.replace(';','')
        
        except: 
            value = 0

        dict_items[name] = value

    data = dict_items["originalData"]

    # Formatting
    data = json.loads(data)
    df_data = pd.DataFrame.from_dict(data)

    df_data["field_name"] = df_data["field_name"].str.split(">").str[1]
    df_data["field_name"] = df_data["field_name"].str.split("<").str[0]
    df_data = df_data.drop(["popup_icon"], axis = 1)
    df_data = df_data.rename(columns = {'field_name':'Date'})
    df_data.index = df_data["Date"]
    df_data = df_data.drop(["Date"], axis = 1)
    df_data = df_data.T
    df_data = df_data.reset_index()
    df_data = df_data.rename(columns = {'index':'Date'})
    df_data = df_data.sort_values(by=["Date"])

    return df_data



# Initializtion
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import datetime
import warnings
import seaborn as sns
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

########
# Y AXIS
########

# Get Historical returns dataset
data = pd.read_csv('../Database/master__netinc.csv')
data["TTM Net Income"] = data["TTM Net Income"].mul(1000000)
data['Prev TTM Net Income'] = data['TTM Net Income'].shift(4)
data = data[data['TTM Net Income']>= 0]
data = data[data['Prev TTM Net Income'] >= 0]
data['chg'] = data['TTM Net Income']/data['Prev TTM Net Income'] - 1
data.dropna(how='any', inplace=True)
data = data.rename(columns={"Date":"original date", "Ticker":"ticker"})

# while we test
data = data.sample(50)

# Smooth the data
data['adj chg'] = data['chg']
data.loc[data['chg'] >= 0.33, 'adj chg'] = 0.33
data.loc[data['chg'] <= -0.21, 'adj chg'] = -0.21

########
# X AXIS
########

# Get market cap for each comp at original date 

link_db = "C:/Users/eirik/Codebase/Database"
link_mt = "https://www.macrotrends.net/stocks/charts"

sharelist = []
tickerlist = data["ticker"].tolist()
for i in tqdm(range(len(data))):

    ticker = tickerlist[i] 
    master_tickers = pd.read_csv(f"{link_db}/master__tickers.csv")
    meta_stock = master_tickers.loc[master_tickers['ticker'] == ticker]
    comp = meta_stock["comp_name"].values
    comp = str(comp).translate(str.maketrans("", "", "[]'\""))
    comp_b = meta_stock["comp_name_2"].values
    link_is = f"{link_mt}/{ticker}/{comp}/income-statement?freq=Q"

    df_is = scrape_mt(link_is)
    df_is["shares"] = pd.to_numeric(df_is["Shares Outstanding"], errors = "coerce").fillna(0).mul(1000000)
    df_is["ticker"] = ticker
    sharelist.append(df_is)

df_shares = pd.concat(sharelist, axis = 0)

df_shares = df_shares[["Date","ticker","shares"]]
df_shares = df_shares.rename(columns={"Date":"original date"})

########
# VALUES
########

# Get prices for dataset
    # Price t=0
tickerlist = data['ticker'].tolist()
datelist = data['original date'].tolist()

datalist = []
datalist_6m = []
datalist_1y = []
datalist_2y = []

def dateadd(dato,add_days):
    return datetime.datetime.strptime(dato,'%Y-%m-%d') + datetime.timedelta(days = add_days)

for i in tqdm(range(len(data))):

    ticker = tickerlist[i]
    date = datelist[i]
    date_base = date
    date_plusone = dateadd(date,1)
    date_plustwo = dateadd(date,2)
    date_plusthree = dateadd(date,3)
    date_plusfour = dateadd(date,4)

    # Some dates do not have data, so we need to check 
    # surrounding dates as well (4 dates should suffice)
    df_price = yf.download(ticker, start = date_base, end = date_plusone)['Adj Close'].values
    if len(df_price) == 0:
        df_price = yf.download(ticker, start = date_plusone, end = date_plustwo)['Adj Close'].values
    if len(df_price) == 0:
        df_price = yf.download(ticker, start = date_plustwo, end = date_plusthree)['Adj Close'].values
    if len(df_price) == 0:
        df_price = yf.download(ticker, start = date_plusthree, end = date_plusfour)['Adj Close'].values
    
    row = ([[ticker, date, df_price,]])
    df_row= (
        pd.DataFrame(
            row, 
            columns = ["ticker", "original date", "price"] 
        )
    )
    

    datalist.append(df_row)


df_data = pd.concat(datalist, axis = 0)



    # Price t = 6m
for i in tqdm(range(len(data))):

    ticker = tickerlist[i]
    date = datelist[i]

    date_base = dateadd(date, 180)
    date_plusone = dateadd(date,181)
    date_plustwo = dateadd(date,182)
    date_plusthree = dateadd(date,183)
    date_plusfour = dateadd(date,184)

    # Some dates do not have data, so we need to check 
    # surrounding dates as well (4 dates should suffice)
    df_price_6m = yf.download(ticker, start = date_base, end = date_plusone)['Adj Close'].values
    if len(df_price_6m) == 0:
        df_price_6m = yf.download(ticker, start = date_plusone, end = date_plustwo)['Adj Close'].values
    if len(df_price_6m) == 0:
        df_price_6m = yf.download(ticker, start = date_plustwo, end = date_plusthree)['Adj Close'].values
    if len(df_price_6m) == 0:
        df_price_6m = yf.download(ticker, start = date_plusthree, end = date_plusfour)['Adj Close'].values
    
    row = ([[ticker, date, date_base, df_price_6m,]])
    df_row= (
        pd.DataFrame(
            row, 
            columns = ["ticker", "original date", "date", "price"] 
        )
    )

    datalist_6m.append(df_row)

df_data_6m = pd.concat(datalist_6m, axis = 0)

    # Price t = 1Y
for i in tqdm(range(len(data))):

    ticker = tickerlist[i]
    date = datelist[i]

    date_base = dateadd(date, 365)
    date_plusone = dateadd(date,366)
    date_plustwo = dateadd(date,367)
    date_plusthree = dateadd(date,368)
    date_plusfour = dateadd(date,369)

    # Some dates do not have data, so we need to check 
    # surrounding dates as well (4 dates should suffice)
    df_price_1y = yf.download(ticker, start = date_base, end = date_plusone)['Adj Close'].values
    if len(df_price_1y) == 0:
        df_price_1y = yf.download(ticker, start = date_plusone, end = date_plustwo)['Adj Close'].values
    if len(df_price_1y) == 0:
        df_price_1y = yf.download(ticker, start = date_plustwo, end = date_plusthree)['Adj Close'].values
    if len(df_price_1y) == 0:
        df_price_1y = yf.download(ticker, start = date_plusthree, end = date_plusfour)['Adj Close'].values
    
    row = ([[ticker, date, date_base, df_price_1y,]])
    df_row= (
        pd.DataFrame(
            row, 
            columns = ["ticker", "original date", "date", "price"] 
        )
    )

    datalist_1y.append(df_row)

df_data_1y = pd.concat(datalist_1y, axis = 0)

    # Price t = 2Y
for i in tqdm(range(len(data))):

    ticker = tickerlist[i]
    date = datelist[i]

    date_base = dateadd(date, 730)
    date_plusone = dateadd(date,731)
    date_plustwo = dateadd(date,732)
    date_plusthree = dateadd(date,733)
    date_plusfour = dateadd(date,734)

    # Some dates do not have data, so we need to check 
    # surrounding dates as well (4 dates should suffice)
    df_price_2y = yf.download(ticker, start = date_base, end = date_plusone)['Adj Close'].values
    if len(df_price_2y) == 0:
        df_price_2y = yf.download(ticker, start = date_plusone, end = date_plustwo)['Adj Close'].values
    if len(df_price_2y) == 0:
        df_price_2y = yf.download(ticker, start = date_plustwo, end = date_plusthree)['Adj Close'].values
    if len(df_price_2y) == 0:
        df_price_2y = yf.download(ticker, start = date_plusthree, end = date_plusfour)['Adj Close'].values
    
    row = ([[ticker, date, date_base, df_price_2y,]])
    df_row= (
        pd.DataFrame(
            row, 
            columns = ["ticker", "original date", "date", "price"] 
        )
    )

    datalist_2y.append(df_row)

df_data_2y = pd.concat(datalist_2y, axis = 0)

# Create massive dataset
df_research = pd.merge(
    data,
    df_data,
    on = ['ticker','original date'],
)

# add 6m data
df_research = pd.merge(
    df_research,
    df_data_6m,
    on = ['ticker','original date'],
    suffixes = ('','_6m')
)

# add 1y data
df_research = pd.merge(
    df_research,
    df_data_1y,
    on = ['ticker','original date'],
    suffixes = ('','_1y')
)

# add 2y data
df_research = pd.merge(
    df_research,
    df_data_2y,
    on = ['ticker','original date'],
    suffixes = ('','_2y')
)

# add Shares
df_research = pd.merge(
    df_research,
    df_shares,
    on = ["ticker","original date"]
)

df_research['return_6m'] = df_research["price_6m"]/df_research["price"]-1
df_research['return_1y'] = df_research["price_1y"]/df_research["price"]-1
df_research['return_2y'] = df_research["price_2y"]/df_research["price"]-1
df_research['market_cap'] = df_research["price"] * df_research["shares"]
df_research['ni_to_mcap_ratio'] = df_research['TTM Net Income']/df_research['market_cap']

sz = df_research["ni_to_mcap_ratio"].size-1
df_research['x_percentile'] = df_research['ni_to_mcap_ratio'].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)

df_research["ni_to_mcap_ratio"] = pd.to_numeric(df_research["ni_to_mcap_ratio"], errors = 'coerce')
df_research["return_6m"] = pd.to_numeric(df_research["return_6m"], errors = 'coerce')


# X Bins
bin_edges = list(range(0, 100, 20))  # Bins of width 20 from 0 to 100
df_research['x'] = pd.cut(df_research['x_percentile'], bins=bin_edges, labels=False, right=False)

# Y Bins
df_research["adj chg"] = 100*df_research["adj chg"]
y_bin_edges = list(range(-21,33,4))
df_research['y'] = pd.cut(df_research['adj chg'], bins = y_bin_edges, labels = False, right = False)

# Post Scriptum

# Activate this function to run Module 1
# m1()
