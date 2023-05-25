"""
@author: eirik
@date: 2023-05-12

Research doc for "Impact of relative earnings on volatility and returns in equities"
"""

# --------------------------------------
# Module 1: Distribution of ttm earnings
# --------------------------------------

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
    data[''] = data['chg']
    data.loc[data['chg'] >= 0.32, ''] = 0.32
    data.loc[data['chg'] <= -0.20, ''] = -0.20

    sns.displot(data, x= '', kde = True, binwidth = 0.04)
    plt.show()

    print(data.head(50))
    print(data.info())


# -----------------------------------
# Module 2: Collection of master data
# -----------------------------------

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



def m2():

    # Initializtion
    import sys, os
    import pandas as pd
    import yfinance as yf
    from tqdm import tqdm
    import datetime
    import warnings
    import time
    import datetime
    from datetime import timedelta
    
    warnings.filterwarnings("ignore")

    # Globals
    path_DB =  'C:/Users/eirik/Codebase/Database/'
    time_start = datetime.datetime.now()

    # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__

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
    
    # Remove outlier data
    data['original date'] = pd.to_datetime(data['original date'])
    data = data.loc[data['original date'] <= datetime.datetime.today() - datetime.timedelta(days = 730)] 
    data = data[data['chg'] >= -.5]
    data = data[data['chg'] <= .5]
    
    # while we test
    # data = data.sample(10)

    
    ########
    # X AXIS
    ########

    # Get market cap for each comp at original date 
    link_db = "C:/Users/eirik/Codebase/Database"
    link_mt = "https://www.macrotrends.net/stocks/charts"
    
    uniquetickers = data['ticker'].unique()
    sharelist = []
    tickerlist = uniquetickers.tolist()
    for i in tqdm(range(len(data))):
        
        try:
            ticker = tickerlist[i] 
            master_tickers = pd.read_csv(f"{link_db}/master__tickers.csv")
            meta_stock = master_tickers.loc[master_tickers['ticker'] == ticker]
            comp = meta_stock["comp_name"].values
            comp = str(comp).translate(str.maketrans("", "", "[]'\""))
            link_is = f"{link_mt}/{ticker}/{comp}/income-statement?freq=Q"

            df_is = scrape_mt(link_is)
            df_is["shares"] = pd.to_numeric(df_is["Shares Outstanding"], errors = "coerce").fillna(0).mul(1000000)
            df_is["ticker"] = ticker
            sharelist.append(df_is)
            time.sleep(0.5)
        except:
            pass

    df_shares = pd.concat(sharelist, axis = 0)

    df_shares = df_shares[["Date","ticker","shares"]]
    df_shares = df_shares.rename(columns={"Date":"original date"})


    ########
    # VALUES 
    ########

    # Get prices for dataset
    df_prices = pd.read_csv(f'{path_DB}master__price.csv')

    # Ok now you can talk
    enablePrint()

    # Add 3 dummy dates to join on
    data['original date'] = pd.to_datetime(data['original date'])
    data['original date t1'] = data['original date'] + timedelta(days=1)
    data['original date t2'] = data['original date'] + timedelta(days=2)
    data['original date t3'] = data['original date'] + timedelta(days=3)

    data['6m'] = data['original date'] + timedelta(days = 182)
    data['6m t1'] = data['original date'] + timedelta(days=183)
    data['6m t2'] = data['original date'] + timedelta(days=184)
    data['6m t3'] = data['original date'] + timedelta(days=185)
    
    data['1y'] = data['original date'] + timedelta(days = 365)
    data['1y t1'] = data['original date'] + timedelta(days=366)
    data['1y t2'] = data['original date'] + timedelta(days=367)
    data['1y t3'] = data['original date'] + timedelta(days=368)
   
    data['2y'] = data['original date'] + timedelta(days = 730)
    data['2y t1'] = data['original date'] + timedelta(days=731)
    data['2y t2'] = data['original date'] + timedelta(days=732)
    data['2y t3'] = data['original date'] + timedelta(days=733)

    df_prices['original date'] = pd.to_datetime(df_prices['Date'])
    df_prices['price'] = df_prices['Adj Close']

    df_shares['original date'] = pd.to_datetime(df_shares['original date'])

    # Create dataset (god forgive me for this code but I dont know how to do 
    # it differently
   
    # t = 0
    df_research = pd.merge(
        data,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','original date'],
        right_on = ['ticker','original date'],
        suffixes = ['','_a1'] # pyright: ignore

    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','original date t1'],
        right_on = ['ticker','original date'],
        suffixes = ['','_a2'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','original date t2'],
        right_on = ['ticker','original date'],
        suffixes = ['','_a3'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','original date t3'],
        right_on = ['ticker','original date'],
        suffixes = ['','_a4'] # pyright: ignore
    )

    df_research['price'] = df_research.price.combine_first(df_research.price_a2)
    df_research['price'] = df_research.price.combine_first(df_research.price_a3)
    df_research['price'] = df_research.price.combine_first(df_research.price_a4)

    df_research.drop(
        ['original date_a2','original date_a3','original date_a4',
         'price_a2','price_a3','price_a4'],
        axis = 1,
        inplace = True
    )

    # t = 6m
    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','6m'],
        right_on = ['ticker','original date'],
        suffixes = ['','_6m'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','6m t1'],
        right_on = ['ticker','original date'],
        suffixes = ['','_6m_a1'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','6m t2'],
        right_on = ['ticker','original date'],
        suffixes = ['','_6m_a2'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','6m t3'],
        right_on = ['ticker','original date'],
        suffixes = ['','_6m_a3'] # pyright: ignore
    )

    df_research['price'] = df_research.price_6m.combine_first(df_research.price_6m_a1)
    df_research['price'] = df_research.price_6m.combine_first(df_research.price_6m_a2)
    df_research['price'] = df_research.price_6m.combine_first(df_research.price_6m_a3)

    df_research.drop(
        ['original date_6m_a1','original date_6m_a2','original date_6m_a3',
        'price_6m_a1','price_6m_a2','price_6m_a3'],
        axis = 1,
        inplace = True
    )

    # t = 1y
    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','1y'],
        right_on = ['ticker','original date'],
        suffixes = ['','_1y'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','1y t1'],
        right_on = ['ticker','original date'],
        suffixes = ['','_1y_a1'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','1y t2'],
        right_on = ['ticker','original date'],
        suffixes = ['','_1y_a2'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','1y t3'],
        right_on = ['ticker','original date'],
        suffixes = ['','_1y_a3'] # pyright: ignore
    )

    df_research['price'] = df_research.price_1y.combine_first(df_research.price_1y_a1)
    df_research['price'] = df_research.price_1y.combine_first(df_research.price_1y_a2)
    df_research['price'] = df_research.price_1y.combine_first(df_research.price_1y_a3)

    df_research.drop(
        ['original date_1y_a1','original date_1y_a2','original date_1y_a3',
        'price_1y_a1','price_1y_a2','price_1y_a3'],
        axis = 1,
        inplace = True
    )
   
    # t = 2y
    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','2y'],
        right_on = ['ticker','original date'],
        suffixes = ['','_2y'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','2y t1'],
        right_on = ['ticker','original date'],
        suffixes = ['','_2y_a1'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','2y t2'],
        right_on = ['ticker','original date'],
        suffixes = ['','_2y_a2'] # pyright: ignore
    )

    df_research = pd.merge(
        df_research,
        df_prices[['ticker','original date','price']],
        how = 'left',
        left_on = ['ticker','2y t3'],
        right_on = ['ticker','original date'],
        suffixes = ['','_2y_a3'] # pyright: ignore
    )

    df_research['price'] = df_research.price_2y.combine_first(df_research.price_2y_a1)
    df_research['price'] = df_research.price_2y.combine_first(df_research.price_2y_a2)
    df_research['price'] = df_research.price_2y.combine_first(df_research.price_2y_a3)

    df_research.drop(
        ['original date_2y_a1','original date_2y_a2','original date_2y_a3',
        'price_2y_a1','price_2y_a2','price_2y_a3'],
        axis = 1,
        inplace = True
    )

    # Add shares
    df_research = pd.merge(
        df_research,
        df_shares,
        on = ["ticker","original date"]
    )

    df_research.drop(
        [
            'original date t1', 'original date t2', 'original date t3',
            '6m t1', '6m t2', '6m t3',
            '1y t1', '1y t2', '1y t3',
            '2y t1', '2y t2', '2y t3',
            'original date_6m', 'original date_1y', 'original date_2y',
        ],
        axis = 1,
        inplace = True
    )

    print(df_research.head(10))
    print(df_research.info())

    # Data Type fixes
    df_research['return_6m'] = df_research["price_6m"]/df_research["price"]-1
    df_research['return_1y'] = df_research["price_1y"]/df_research["price"]-1
    df_research['return_2y'] = df_research["price_2y"]/df_research["price"]-1
    df_research['mcap'] = df_research["price"] * df_research["shares"]
    df_research['ni_to_mcap_ratio'] = df_research['TTM Net Income']/df_research['mcap']

    sz = df_research["ni_to_mcap_ratio"].size-1
    df_research['x_percentile'] = df_research['ni_to_mcap_ratio'].rank(method='max').apply(lambda x: 100.0*(x-1)/sz)
    df_research["ni_to_mcap_ratio"] = pd.to_numeric(df_research["ni_to_mcap_ratio"], errors = 'coerce')

    # Save the results to run on a different day blud
    df_research.to_csv(f'{path_DB}research__earningsheatmap.csv', index = False)

    filesize = len(df_research)
    time_duration = datetime.datetime.now() - time_start  
    print(f'Wrote {filesize} records')
    print(f'Runtime: {time_duration}')


    # X Bins
#     df_research['x'] = pd.cut(
#         df_research['x_percentile'],
#         bins = 5,
#         labels = False,
#         right = True,
#         include_lowest= True
#     )
#
#     # Y Bins
#     df_research["chg"] = 100*df_research["chg"]
#     df_research['y'] = pd.cut(
#         df_research['chg'], 
#         bins = 10,
#         labels = False, 
#         right = True,
#         include_lowest = True
#     )


     

# Activate this function to run Module 1
# m1()

# Activate this function to run Module 2
# m2()

# ------------------ 
# Module 3: Analysis
# ------------------

# Initializtion
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt

path_DB =  'C:/Users/eirik/Codebase/Database/'
df_research = pd.read_csv(f'{path_DB}research__earningsheatmap.csv')
df_research.drop_duplicates(inplace=True)
df_research.info()

# X Bins
df_research['x'] = pd.cut(
    df_research['x_percentile'],
    bins = 5,
    labels = False,
    right = True,
    include_lowest= True
)

# Y Bins
df_research["chg"] = 100*df_research["chg"]
df_research['y'] = pd.cut(
    df_research['chg'], 
    bins = 10,
    labels = False, 
    right = True,
    include_lowest = True
)

def analysis_6m():
# t = 6m
    df_research_6m      = df_research[['x','y','return_6m']].dropna()
    df_heatmap6m_vol    = df_research.groupby(['x','y'])[['return_6m']].std().reset_index() # pyright: ignore
    df_heatmap6m_counts = df_research.groupby(['x','y'])[['return_6m']].count().reset_index()
    df_heatmap6m        = df_research.groupby(['x','y'])[['return_6m']].mean().reset_index()

    df_heatmap = df_heatmap6m.pivot('y','x','return_6m')
    df_heatmap_counts = df_heatmap6m_counts.pivot('y','x','return_6m')
    df_heatmap_vol = df_heatmap6m_vol.pivot('y','x','return_6m')

# Stats
    df_heatmap6m_stats_x = df_research.groupby(['x'])[['ni_to_mcap_ratio']].max()
    print(df_heatmap6m_stats_x.head(10))

    df_heatmap6m_stats_y = df_research.groupby(['y'])[['chg']].max()
    print(df_heatmap6m_stats_y.head(10))

# Plots
    fig, axs = plt.subplots(ncols=3)

    sns.heatmap(
        df_heatmap, 
        annot = True, 
        cmap = sns.diverging_palette(15, 150, s=70, as_cmap=True),
        square = True,
        ax = axs[0]
    ).set( # pyright: ignore
        title = '6 Month Returns'
    )

    sns.heatmap(
        df_heatmap_counts, 
        annot = True, 
        cmap = sns.diverging_palette(15, 150, s=70, as_cmap=True),
        square = True,
        ax = axs[1]
    ).set( # pyright: ignore
        title = '# of Data Points'
    )

    sns.heatmap(
        df_heatmap_vol,
        annot = True,
        cmap = sns.diverging_palette(150, 15, s=70, as_cmap=True),
        square = True,
        robust = True,
        ax = axs[2]
    ).set( # pyright: ignore
        title = 'Standard Deviation'
    )

    plt.show()


# t = 1y
df_research_1y      = df_research[['x','y','return_1y']].dropna()
df_heatmap1y_vol    = df_research.groupby(['x','y'])[['return_1y']].std().reset_index() # pyright: ignore
df_heatmap1y_counts = df_research.groupby(['x','y'])[['return_1y']].count().reset_index()
df_heatmap1y        = df_research.groupby(['x','y'])[['return_1y']].mean().reset_index()

df_heatmap = df_heatmap1y.pivot('y','x','return_1y')
df_heatmap_counts = df_heatmap1y_counts.pivot('y','x','return_1y')
df_heatmap_vol = df_heatmap1y_vol.pivot('y','x','return_1y')

# Stats
df_heatmap1y_stats_x = df_research.groupby(['x'])[['ni_to_mcap_ratio']].max()
print(df_heatmap1y_stats_x.head(10))

df_heatmap1y_stats_y = df_research.groupby(['y'])[['chg']].max()
print(df_heatmap1y_stats_y.head(10))

# Plots
fig, axs = plt.subplots(ncols=3)

sns.heatmap(
    df_heatmap, 
    annot = True, 
    cmap = sns.diverging_palette(15, 150, s=70, as_cmap=True),
    square = True,
    ax = axs[0]
).set( # pyright: ignore
    title = '1 Year Returns'
)

sns.heatmap(
    df_heatmap_counts, 
    annot = True, 
    cmap = sns.diverging_palette(15, 150, s=70, as_cmap=True),
    square = True,
    fmt = '',
    ax = axs[1]
).set( # pyright: ignore
    title = '# of Data Points'
)

sns.heatmap(
    df_heatmap_vol,
    annot = True,
    cmap = sns.diverging_palette(150, 15, s=70, as_cmap=True),
    square = True,
    robust = True,
    ax = axs[2]
).set( # pyright: ignore
    title = 'Standard Deviation'
)

plt.show()
