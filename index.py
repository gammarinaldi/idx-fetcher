import yfinance as yf
import watchlist
import concurrent.futures
import traceback
import random
import pandas as pd
import glob
import csv
import requests
import json
import time
import os

from dotenv import load_dotenv
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from time import sleep

load_dotenv()
proxy_rotator_url = os.getenv('PROXY_ROTATOR_URL')
proxy_rotator_key = os.getenv('PROXY_ROTATOR_KEY')
dir_path = os.getenv('DIR_PATH')

def proxy_rotator():
    params = dict(apiKey=proxy_rotator_key)
    resp = requests.get(url=proxy_rotator_url, params=params)
    resp_text = json.loads(resp.text)
    return resp_text['proxy']

def fetch(symbol):
    # sleep(random.randint(3, 5))
    try:
        proxy = proxy_rotator()
        # proxy = "none"
        
        # fetch quote
        df_list = list()
        df = yf.download(symbol, period="1d", group_by="ticker", proxy=proxy)
        df['Ticker'] = symbol.replace(".JK", "")
        df_list.append(df)            

        # combine all dataframes into a single dataframe
        df = pd.concat(df_list)
        
        # save to csv
        df[["Ticker", "Open", "High", "Low", "Close", "Volume"]].to_csv(f"{dir_path}\\csv\\{symbol}.csv")

        print(f"Fetching {symbol} with proxy {proxy}: success!")
    except Exception as error:
        print(f"Fetching {symbol} with proxy {proxy}: failed!")
        write_to_csv(symbol, f"{dir_path}\\failed.csv")
        print(error)

def write_to_csv(data, file_name):
    if data in watchlist.list:
        row = [data]
    else:
        item = data.split(",")
        symbol = "IHSG" if item[0] == "JKSE" else item[0]
        date = item[1]
        o = item[2]
        h = item[3]
        l = item[4]
        c = item[5]
        v = item[7]
        row = [symbol, date, o, h, l, c, v] #the data
    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f) #this is the writer object
        writer.writerow(row) #this is the data    

def executor_submit(executor, stock_list):
    return {executor.submit(fetch, symbol): symbol for symbol in stock_list}

def fetch_async(stock_list):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_user = executor_submit(executor, stock_list)
        for future in concurrent.futures.as_completed(future_to_user):
            try:
                if future.result() != None:
                    print("Async result error")
                    print(future.result())
            except Exception as error:
                print("Exeption error occured:")
                print(error)
                print(traceback.format_exc())

def retry_fetch():
    path = f"{dir_path}\\failed.csv"
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        if is_empty_csv(path) == False:
            stock_list = []
            for row in csvreader: stock_list.append(row)
            fetch_async(stock_list)
        else: 
            print("Nothing to retry")

def merge_csv():
    # Merge all emiten data
    files = glob.glob(f"{dir_path}\\csv\\*.csv")
    df = pd.concat((pd.read_csv(f, header = 0) for f in files))
    # now = dt.now().strftime('%Y%m%d')
    df.to_csv(f"{dir_path}\\merged\\result.csv")

def is_empty_csv(path):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for i, _ in enumerate(reader):
            if i:  # Found the second row
                return False
    return True


if __name__ == '__main__':
    print("Start IDX updater...")
    t1 = time.time()

    fetch_async(watchlist.list)
    # fetch("GOTO")
    # fetch(all_stock.list2)
    # retry_fetch()
    merge_csv()

    t2 = time.time()
    diff = t2 -t1
    print("Elapsed times: " + str(round(diff, 2)))