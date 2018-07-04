import pandas as pd
import datetime

from ta import *  #https://github.com/bukosabino/ta
import numpy as np
import urllib.request

import json
import time

import datetime

from random import randint

"""
Copyright 2018 Görkem Göknar
Educational community usage only
Educational Community License, Version 2.0 (ECL-2.0)
"""


"""
DISCLAIMER REGARDING FORECAST AND PREDICTIONS:
Financial Results and coding work in this project are done for educational purposes only. Actual stock performance may differ from those predicted by this work. No guarantee is presented or implied as accuracy of forecasts contained. I am not responsible for any financial loss made using this report and code provided.
"""



def get_stock_from_url(url):
    #get stock 

    contents = urllib.request.urlopen(url).read()
    return json.loads(contents)

def get_daily_stock_as_pd(stock):
    labels = ['Stock', 'Date', 'High', 'Low', 'Volume', 'Close']
     
    print("Stock: {}".format(stock["Index"]))
    stock_daily_price = []
    
    for daily_vals in stock["Data"]["ohlc"]:
    
        day_as_timestamp = daily_vals[0]
        daily_end_price = daily_vals[1]
        daily_high = stock["Data"]["Tooltips"]["Yüksek"][str(day_as_timestamp)]
        daily_low = stock["Data"]["Tooltips"]["Düşük"][str(day_as_timestamp)]
        daily_vol = stock["Data"]["Tooltips"]["Hacim"][str(day_as_timestamp)]
        
        stock_daily_price.append( [stock["Index"], pd.to_datetime(day_as_timestamp/1000, unit='s'), daily_high, daily_low, daily_vol, daily_end_price])
        #print("Date:{0} end: {1:.4f} , high : {2:.4f} , low : {3:.4f} , vol : {4}".format(
        #    convert_to_datetime(day_as_timestamp),daily_end_price, daily_high, daily_low, daily_vol))
        
    return pd.DataFrame.from_records(stock_daily_price, columns=labels)

def get_stock_from_mynet(stock_name,stock_range="yillik_5",type="stock"):
    #fetches stock data from mynet ajax
    
    url_base = "http://finans.mynet.com" 
    #stock_yillik_10 = url_base + yillik_10
    base_url = "/borsa/ajaxCharts/?type=" + type +"&ticker="
        
    stock_ranges = {
    "haftalik" : base_url + stock_name + "&range=w"   ,
    "aylik_1"   :   base_url + stock_name + "&range=m" , 
    "aylik_3"   :   base_url+ stock_name + "&range=3m"   ,        
    "aylik_6"   :   base_url + stock_name + "&range=6m"  ,     
    "yillik_1"  :   base_url + stock_name + "&range=y" ,      
    "yillik_3"  :   base_url + stock_name + "&range=3y"  ,       
    "yillik_5"  :   base_url + stock_name + "&range=5y"   ,      
    "yillik_10" :   base_url + stock_name + "&range=10y"  ,
    
    }
    url = url_base + stock_ranges[stock_range]
    print(url)
    stock= get_stock_from_url(url)
    
    return get_daily_stock_as_pd(stock)


def get_daily_exchange_as_pd(stock):
    
    labels = ['Exchange', 'Date', 'High', 'Low', 'Close']
    
    
    print("Stock: {}".format(stock["Index"]))
    stock_daily_price = []
    
    for daily_vals in stock["Data"]["ohlc"]:
    
        day_as_timestamp = daily_vals[0]
        daily_end_price = daily_vals[1]
        daily_high = stock["Data"]["Tooltips"]["Yüksek"][str(day_as_timestamp)]
        daily_low = stock["Data"]["Tooltips"]["Düşük"][str(day_as_timestamp)]
        
        stock_daily_price.append( [stock["Index"],pd.to_datetime(day_as_timestamp/1000, unit='s'), daily_high, daily_low,  daily_end_price])
        #print("Date:{0} end: {1:.4f} , high : {2:.4f} , low : {3:.4f} , vol : {4}".format(
        #    convert_to_datetime(day_as_timestamp),daily_end_price, daily_high, daily_low, daily_vol))
        
    return pd.DataFrame.from_records(stock_daily_price, columns=labels)


def get_exchange_from_mynet(exchange_name,exchange_range="yillik_5", no_extra=False):
    #EUR , USD, GBP, JPY, CHF, RUB, CNY, BHD, BRL
    
    url_base = "http://finans.mynet.com" 
    
    exchange_ranges = {
    "haftalik" : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=w"   ,
    "aylik_1"   : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=m"   ,
    "aylik_3"   : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=3m"   ,   
    "aylik_6"   : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=6m"   ,
    "yillik_1"  : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=y"   ,
    "yillik_3"  : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=3y"   ,  
    "yillik_5"  : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=5y"   , 
    "yillik_10"  : "/doviz/ajaxCharts/?doviz=" + exchange_name + "&market=freeMarket&period=10y"   , 
        
    }
    url = url_base + exchange_ranges[exchange_range]
    
    print(url)
    exchange= get_stock_from_url(url)
    
    return get_daily_exchange_as_pd(exchange)

def convert_to_datetime(timestamp):
    import time
    return time.strftime("%d/%m/%Y", time.gmtime(timestamp / 1000.0))



def season(date, hemisphere=None):
    ''' date is a datetime object
        hemisphere is either 'north' or 'south', dependent on long/lat.
    '''
    if hemisphere is None:
        hemisphere = "North"
        
    md = date.month * 100 + date.day

    if ((md > 320) and (md < 621)):
        s = 0 #spring
    elif ((md > 620) and (md < 923)):
        s = 1 #summer
    elif ((md > 922) and (md < 1223)):
        s = 2 #fall
    else:
        s = 3 #winter

    if hemisphere != 'north':
        if s < 2:
            s += 2 
        else:
            s -= 2

    return s

def date_features(s):
    
    s['Day_of_week'] = s["Date"].dayofweek
   
    #num days passed from last open day, assume sorted ascending
    s["Days_from_last_open"] = s['Date'] - s['Date'].shift(1)
    s['Days_from_last_open'] = s['Days_from_last_open'].astype('timedelta64[D]')
    #s['month'] = s["Date"].month
    #s['week'] = s["Date"].week
    #s['season'] = locale.format("%.1f", s['size'] / 1024.0 ** 3, grouping=True) + ' GB'
    return s


def get_stock_features(df, has_volume=False,is_target=False, hasOpen = False,buy_threshold = 0.000001 ):

    if is_target:
        df['Day_of_week'] = df["Date"].dt.dayofweek   
        #num days passed from last open day, assume sorted ascending
        df["Days_from_last_open"] = df['Date'] - df['Date'].shift(1)
        df['Days_from_last_open'] = df['Days_from_last_open'].astype('timedelta64[D]')

    
    #replace any 0 with front fill
    df['High'].replace(to_replace=0, method='ffill',inplace=True)
    df['Low'].replace(to_replace=0, method='ffill',inplace=True)
    df['Close'].replace(to_replace=0, method='ffill',inplace=True)


    
    prev_day_close= df["Close"].shift(1)
    
    df["High_Close_perc"]= (df["Close"] - df["High"])/df["Close"]*100
    df["Close_Low_perc"]= (df["Close"] - df["Low"])/df["Close"]*100
    df["Prev_Day_Close_perc"]= (df["Close"] - prev_day_close)/df["Close"]*100
    
    if is_target:
        days_forward = -1
        Next_day_value_percent = ( df["Close"].shift(days_forward) - df["Close"])/df["Close"]
        #df["Next_day_prediction"] = Next_day_value_percent.apply(lambda x: 2if x>buy_threshold else (0 if x<(sell_threshold) else 1))
        df["Next_day_prediction"] = Next_day_value_percent.apply(lambda x: 1 if x>buy_threshold else 0)
        df["Next_day_buy_price"]= df["Close"] #buy next day at todays close
    if has_volume:
        Prev_Day_Volume= df["Volume"].shift(1)
        df["Prev_Day_Volume_Change"]= (df["Volume"] - Prev_Day_Volume)/df["Volume"]*100

    #df.drop("Next_day_value_percent",inplace=True)
    
    return df

def generate_ta_features(df,period=5,period_slow=10, no_extra=False):

    high = df["High"]
    low = df["Low"]
    close= df["Close"]
    
    
    #RSI indicator
    ##RSI > 70 --> overvalued -> sell it 
    ##RSI < 30 --> undervalued --> buy it 
    
    df["RSI"]= momentum.rsi(df["Close"], n=period, fillna=False)
    #rsi_threshold = 0.05
    #df["RSI_Prev"] = (df["RSI"] - df["RSI"].shift(1)) > rsi_threshold
    random_factor = 0
    
    def rsi_indicator(x):
        #x 0 -> rsi
        #will use some randomness
        if x[0] >= (70 + randint(-random_factor, random_factor)) :
            return 0 #sell
        
        if x[0] < (30 + randint(-random_factor, random_factor)) :
            return 2 #buy
        else:
            return 1 # hold 
        
        #else
        return 1
        
    df["RSI_indicator"] = df[["RSI"]].apply(rsi_indicator, axis=1)
    ##check rsi, if it is below 30 and higher than 1 days ago value, time to buy
    ##if it is above 70 do asell
    
    #done with original RSI
    df.drop("RSI",axis=1,inplace=True)
    
    
    if not no_extra:
	    df["ATR"]= volatility.average_true_range(df["High"], df["Low"], df["Close"], n=period, fillna=False)
	    
	    #indicators either 0 or 1
	    df["DON_HBAND_IND"]= volatility.donchian_channel_hband_indicator(df["Close"], n=period, fillna=False)
	    df["DON_LBAND_IND"] =volatility.donchian_channel_lband_indicator(df["Close"], n=period, fillna=False)
	    
        #keltner band always 0 drop it.
        #df["KLTNER_HBAND_IND"]= volatility.keltner_channel_hband_indicator(df["High"], df["Low"], df["Close"], n=period, fillna=False)
	    #df["KLTNER_LBAND_IND"]= volatility.keltner_channel_lband_indicator(df["High"], df["Low"], df["Close"], n=period, fillna=False)
	    
	    
	    
	    df["ADX"]= trend.adx(df["High"], df["Low"], df["Close"], n=5, fillna=False)

    #CCI Basic Strategy
    #A basic CCI strategy is used to track the CCI for movement above +100, which generates buy signals, and movements below -100, which generates sell or short trade signals. Investors may only wish to take the buy signals, exit when the sell signals occur, and then re-invest when the buy signal occurs again
    #Read more: How traders use CCI (Commodity Channel Index) to trade stock trends | Investopedia https://www.investopedia.com/articles/active-trading/031914/how-traders-can-utilize-cci-commodity-channel-index-trade-stock-trends.asp#ixzz5ImHdHIFM 
    
    df["CCI"]=trend.cci(high, low, close, n=period_slow, c=0.015, fillna=False)
    def cci_indicator(x):
        #using some randomness
        if x[0] >= (100+ randint(-random_factor, random_factor)) :
            return 2 #buy
        if x[0] < (-100+ randint(-random_factor, random_factor)):
            return 0 #sell
        #else hold
        return 1

    df["CCI_indicator"] = df[["CCI"]].apply(cci_indicator, axis=1)
    #done with original CCI
    df.drop("CCI",axis=1,inplace=True)

    df["MACD_Signal"]=trend.macd_signal(close, n_fast=period_slow, n_slow=period_slow*2, n_sign=6, fillna=False)

    df["EMA_F"]=trend.ema_fast(close, n_fast=period, fillna=False)
    df["EMA_Fast_slope"] = np.gradient(df["EMA_F"].values, 1)  #slope!
    df["EMA_F_indicator"] = df["EMA_Fast_slope"].apply(lambda x: 1 if x >=0 else 0 )
    df.drop(["EMA_F","EMA_Fast_slope"],axis=1,inplace=True)

    df["EMA_S"]=trend.ema_slow(close, n_slow=period_slow, fillna=False)
    df["EMA_Slow_slope"] = np.gradient(df["EMA_S"].values, 1)  #slope!
    df["EMA_S_indicator"] = df["EMA_Slow_slope"].apply(lambda x: 1 if x >=0 else 0 )
    df.drop(["EMA_S","EMA_Slow_slope"],axis=1,inplace=True)

    if not no_extra:
        df["MACD"]=trend.macd(close, n_fast=period_slow, n_slow=period_slow*2, fillna=False)
        #Detrended Price Oscillator
        df["DPO"]=trend.dpo(close, n=period_slow, fillna=False)

        #positive when price above dpo
        df["DPO_indicator"] = df["DPO"].apply(lambda x: 1 if x>=0 else 0)
        df.drop("DPO",axis=1,inplace=True)


        df["ICH_A"]=trend.ichimoku_a(high, low, n1=period, n2=period_slow, fillna=False)
        df["ICH_B"]=trend.ichimoku_b(high, low, n2=period_slow*1, n3=period_slow*2, fillna=False)
        
        #DO not use KST as makes drop precisous traing data (50 rows!!)
        #df["KST_SIG"]=trend.kst_sig(close, r1=period, r2=period_slow, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=False)
        df["Trix"]=trend.trix(close, n=period, fillna=False)
        #df["Daily_return"]=others.daily_return(close, fillna=False)
    
    return df

def generate_features_for_df(df,has_volume=False, is_target=False, no_extra=False):

    #first normalize date on days
    df = df.sort_values('Date',ascending=True).reset_index(drop=True)
    df['Date'] =pd.to_datetime(df['Date']).dt.normalize()

    #remove duplicate dates before calculation (duplicate may happen on exchanges)
    df = df.drop_duplicates(subset='Date', keep="last")

    df = get_stock_features(df,has_volume=has_volume, is_target=is_target)
    #remove duplicate dates
    
    df = generate_ta_features(df, no_extra=no_extra)
    return df


def normalize_series(series):
    # Standardize time series data
    from pandas import Series
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from math import sqrt
    # load the dataset and print the first 5 rows
    #series = Series.from_csv('daily-minimum-temperatures-in-me.csv', header=0)
    #print(series.head(1))
    
    
    # Train the Scaler with training data and smooth data
    smoothing_window_size = round(len(series/4))
  

    # prepare data for standardization
    values = series.values
    values = values.reshape((len(values), 1))
    # train the standardization
    #scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaler = scaler.fit(values)
    #print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
    # standardization the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    """
    for i in range(5):
        print(normalized[i])
    """
    # inverse transform and print the first 5 rows
    #inversed = scaler.inverse_transform(normalized)
    #for i in range(5):
    #    print(inversed[i])
        
    return normalized 

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]

def split_train_test(X,y,percent=80, normalize=True):
    #SPLIT TRAIN TEST
    train_percent = percent

    train_len = round(train_percent/100 * len(X))

    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]


    if normalize:
        #Normalization must be done after splitting in order to not include future data in training
        #Normalize each column
        print("normalizing train")

        for column in X_train.columns:
            #print("before norm:",X_train[column][0])  
            X_train[column] = normalize_series( X_train[column] )
            #print("after norm:",X_train[column][0])  
           
        print("normalizing test")
        for column in X_test.columns:
            
            X_test[column] = normalize_series( X_test[column] )
            
    return (X_train,X_test,y_train,y_test)


def simulate_outcome(action_array,real_data_df, close_col="Close_x",monkey_trading=False, input_money = 100, action_money = 20, sell_all=False, buy_all=False, buy_and_hold = False, debug=False, trade_commision= 0.000):
    print("≈"*20) 
    money_on_hand = input_money
    num_stocks = 0

    total_worth = money_on_hand
    
    stock_close_today = 0 
    
    worth_per_day = []
    days = []
    number_of_trades = 0
    action = []
    todays_action = 0
    if trade_commision < 0.0000001: 
        if debug: print("No Commision considered ")
    
    if buy_and_hold:
        print("BUY AND HOLD STRATEGY")
            
        num_stocks = money_on_hand/ real_data_df.iloc[0][close_col]
        money_on_hand = 0
        
        final_day_value = num_stocks * real_data_df.iloc[-1][close_col]
        number_of_trades = 1
        todays_action  = 1
        
        
        
    else:
        print("Standart Trading predictive strategy")
        
        if debug:
            if buy_all:
                print("Buy strategy: Buying All")
            else:
                print("Buy strategy: Buying 1")
            if sell_all:
                print("Sell strategy: Selling All")
            else:
                print("Sell strategy: Selling 1")
    first_date = None
    last_date = None
    
    for idx, val in enumerate(action_array):
        #get date for that id
       
        stock_close_today = real_data_df.iloc[idx][close_col]
        date = real_data_df.iloc[idx]["Date"]
        
        
        if buy_and_hold:
            if first_date is None:
                first_date = date
        
            todays_worth = money_on_hand + num_stocks*stock_close_today
            worth_per_day.append(todays_worth)
            days.append(date)
            todays_action = 1
            action.append(todays_action)
            continue
            
        if monkey_trading:
            val = randint(0, 1)
            
        if val == 1:
            
            ##BUY if we have money
            if money_on_hand >= action_money:
                if buy_all:
                    if first_date is None:
                        first_date = date
        
                    if debug: print("Date: {} buying all , stock_price: {}".format(date,stock_close_today ))
                    num_stocks += (money_on_hand/stock_close_today)
                    money_on_hand = 0
                    number_of_trades += 1
                    todays_action = 1
                else:
                    if debug: print("Date: {} ,buying {} , stock_price: {}".format(date,action_money,stock_close_today))
                    money_on_hand -= action_money* trade_commision
                    money_on_hand -= action_money
                    num_stocks += action_money/stock_close_today
                    #suppose price increased and put it in exchange
                    number_of_trades += 1
                    todays_action = 1

            else:
                if debug: print("no money to buy")
        elif val == 0:
            ##sell
            if num_stocks>0:
                if sell_all:
                    if debug: print("Date: {} selling all , stock_price: {}".format(date,stock_close_today))
                    money_on_hand = num_stocks * stock_close_today
                    money_on_hand -= money_on_hand * trade_commision
                    num_stocks = 0
                    number_of_trades += 1
                    todays_action = -1
                    
                else:
                    if debug: print("Date: {} , selling {} , stock_price: {}".format(date,action_money,stock_close_today))
                    num_stocks -= action_money/stock_close_today
                    money_on_hand += action_money - action_money*trade_commision
                    number_of_trades += 1
                    todays_action = -1
                    
            else:
                if debug: print("no stock in exchange to sell")
                    
        todays_worth = money_on_hand + num_stocks*stock_close_today
        worth_per_day.append(todays_worth)
        days.append(date)
        action.append(todays_action)
        
    last_date = date
    last_day_stock_close = real_data_df.iloc[-1][close_col]
    print("First trade day:",first_date)
    print("Last trade day:",last_date)
    print("Number of trades:",number_of_trades)
    if monkey_trading:
        print("!!!MONKEY TRADING!!! -random action ")
        print("This monkey gain %: ", ((money_on_hand + stock_close_today* num_stocks) - input_money ) / input_money * 100 )
    
    else:
        if debug: print("Input money: ", input_money)
        if debug: print("Final Money in exchange: ",  stock_close_today* num_stocks)
        if debug: print("Number of stocks: ",  num_stocks)
        
        if debug: print("Final Money on hand: ", money_on_hand )
        print("Total money: ",  last_day_stock_close* num_stocks + money_on_hand)
        print("Increase %: ", (( last_day_stock_close* num_stocks + money_on_hand) - input_money ) / input_money * 100 )
    
    print("≈"*20) 
    d = {'Date': days, 'Worth': worth_per_day, 'Action': action}
    df = pd.DataFrame(data=d)

    return df




