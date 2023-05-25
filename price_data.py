
import sys, os
from tqdm import tqdm
import pandas as pd
import yfinance as yf
import datetime 

# The function yf.download creates a ton of print spam that we want to suppress. These two
# functions will give your terminal peace

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# Globals
time_start = datetime.datetime.now()
path_DB =  'C:/Users/eirik/Codebase/Database/'

# Get scrape inputs
df = pd.read_csv(f'{path_DB}master__tickers.csv')
print(len(df))
base_date = '2013-01-01'
tickerlist = df['ticker'].unique().tolist()
pricelist = []

# Make it not spew
blockPrint()

for i in tqdm(range(len(df))):

    # Since we might get booted at any moment, we put in 
    # a TRY to evade getting thrown
    try:
        ticker = tickerlist[i]
        df_price = yf.download(ticker, start = base_date)
        df_price['ticker'] = ticker
        pricelist.append(df_price)
    
    except:
        pass

# Assemble dataset and write to disk
df_price = pd.concat(pricelist, axis = 0)
df_price.to_csv(f'{path_DB}master__price.csv', index=True)

# Enable printing again and print diagnostics
enablePrint()
filesize = len(df_price)
num_tickers = len(df_price['ticker'].unique())
runtime = datetime.datetime.now() - time_start
print(f'Wrote {filesize} records for {num_tickers} equities')
print(f'Runtime: {runtime}')

