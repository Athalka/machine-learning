import requests
from bs4 import BeautifulSoup
import csv 
import pandas as pd
import datetime

"""
Copyright 2018 Görkem Göknar
Educational community usage only
Educational Community License, Version 2.0 (ECL-2.0)
"""


"""
DISCLAIMER REGARDING FORECAST AND PREDICTIONS:
Financial Results and coding work in this project are done for educational purposes only. Actual stock performance may differ from those predicted by this work. No guarantee is presented or implied as accuracy of forecasts contained. I am not responsible for any financial loss made using this report and code provided.
"""


def get_stock_from_as_df(currency_id,currency_name,years=5,interval="Daily",sort_order="ASC",base="www",hasVolume=False):

    base_url = "https://" + base + ".investing.com"  ##base changes date format!! and decimal pointer
    #base_url = "https://uk.investing.com"

    sub_url = "/instruments/HistoricalDataAjax"
    url = base_url + sub_url

    today = datetime.datetime.now()
    end_date= today.strftime('%d/%m/%Y')
    start_date = (today - datetime.timedelta(days=years*365)).strftime('%d/%m/%Y')
    #dax 172

    #currency_id = currency_id #akbank tr 
    #header = currency_name
    #interval = "Daily" #Daily, Weekly, Monthly
    sort_column_by = "date"
    #sort_order = "ASC"   # ASC , DESC
    start_date = "08/07/2013"  #DD/MM/YYYY 
    end_date = "12/06/2018" #DD/MM/YYYY


    data = {
        "action":"historical_data",
        "curr_id" : currency_id,
        "st_date": start_date,
        "end_date": end_date,
        "header" : currency_name,
        "interval_sec" : interval,
        "sort_col" : sort_column_by,
        "sort_ord" : sort_order,
        }

    headers= {
                'Origin': 'http://www.investing.com',
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu' +
                ' Chromium/51.0.2704.79 Chrome/51.0.2704.79 Safari/537.36',
                'X-Requested-With': 'XMLHttpRequest'
            }
    #print("currency_id:", currency_id)
    #print(url)
    r = requests.post(url, data=data, headers=headers)

    #print(r.status_code, r.reason)

    if(r.status_code != 200):
        #print("Status is NOT OK abort csv generating")
        return None

    def convert_volume_to_number(x):
        if x.endswith('k'):
            return float(x[:-1])*1000
        elif x.endswith('K'):
            return float(x[:-1])*1000
        elif x.endswith('m'):
            return float(x[:-1])*1000000
        elif x.endswith('M'):
            return float(x[:-1])*1000000
        elif x.endswith('b'):
            return float(x[:-1])*1000000000
        elif x.endswith('B'):
            return float(x[:-1])*1000000000

        elif x.endswith("-"):
            return None
        else:
            return float(x)

    df  = pd.read_html(r.text, attrs={'class': 'genTbl'})[0].dropna(axis=0, thresh=4)
    
    if hasVolume:
        df.rename(columns={'Vol.': 'Volume'}, inplace=True)
        df["Volume"] = df["Volume"].apply(convert_volume_to_number)


    return df

