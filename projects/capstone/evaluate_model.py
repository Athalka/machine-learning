import pandas as pd
import numpy as np

import capstone_module
import importlib
importlib.reload(capstone_module)

from capstone_module import *
from google_trend import *
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

"""
Copyright 2018 Görkem Göknar
Educational community usage only
Educational Community License, Version 2.0 (ECL-2.0)
"""


"""
DISCLAIMER REGARDING FORECAST AND PREDICTIONS:
Financial Results and coding work in this project are done for educational purposes only. Actual stock performance may differ from those predicted by this work. No guarantee is presented or implied as accuracy of forecasts contained. I am not responsible for any financial loss made using this report and code provided.
"""

def predict_stock(stock_df_base,stock_name, train_percent=85, use_all_features = False):
    #Google trend data
    #faiz = interest rate
    #borç = debt
    print("Generating stock features for: ",stock_name)
    stock_df = generate_features_for_df(stock_df_base,has_volume=True, is_target=True)
    

    #Outlier handling 
    def find_zero_and_replace(df,drop=False):
        #replace zero with nearest value
        df= df.replace(to_replace=0.0, method='ffill')
        return df
    def replace_zeroes_in_df(df):
        for col in df.columns:
            df[col] = find_zero_and_replace(df[col])
            
        return df

    if use_all_features:
        #banka = bank , this third keyword will change according to stock, main indicator is people search for bank or banks name for location if needed that day,
        kw_list =["faiz","borç"]
        print("Getting google trends for keywords: {}".format(*kw_list))
        #NOTE that google trends does not show last 2 days, Front fill is used to fill last 2 days
        interest_over_time_df = get_google_trend_data(kw_list=kw_list, years=6, ff_to_today=True)
        interest_over_time_df.drop(["isPartial"],axis=1,inplace=True)
        interest_over_time_df.tail(10)


        #Load BIST 30 Index - Last 5 years
        #bist100_df_base = get_stock_from_mynet("XU100",stock_range="yillik_5",type="index")
        bist30_df_base = get_stock_from_mynet("XU030",stock_range="yillik_5",type="index")
        bist30_df_base.drop(["Stock"],inplace=True,axis=1)
        bist30_df_base = bist30_df_base.sort_values(by='Date').reset_index(drop=True)
        print("BIST30 shape", bist30_df_base.shape)
        print("BIST30 Columns:\n",bist30_df_base.columns)
        #drop outlier
        bist30_df_base.drop( (bist30_df_base.loc[(bist30_df_base["Close"]==0)].index), inplace=True)

        #Foreign Exchange rates
        usdtry_df_base= get_exchange_from_mynet("USD",exchange_range="yillik_5")
        print("USD/TRY Columns:\n",usdtry_df_base.columns)

        usdtry_df_base = replace_zeroes_in_df(usdtry_df_base)


        #eurtry_df= get_exchange_from_mynet("EUR")
        #gbptry_df= get_exchange_from_mynet("GBP")
        #cnytry_df= get_exchange_from_mynet("CNY")
        #bhdtry_df= get_exchange_from_mynet("BHD")

        #Russian Ruble 
        rubtry_df_base= get_exchange_from_mynet("RUB",exchange_range="yillik_5")

        #Brazil Real
        brltry_df_base= get_exchange_from_mynet("BRL",exchange_range="yillik_5")

        #Japanese Yen
        jpytry_df_base= get_exchange_from_mynet("JPY",exchange_range="yillik_5")


        #MAKE FEATURES
        #

       


        ##Generate Index features
        print("Generating Bist30 features")
        #change bist volume  hasVolume to see impact
        bist30_df= generate_features_for_df(bist30_df_base,has_volume=False, no_extra=True)
        bist30_df.drop(["Volume"],axis=1,inplace=True)
        #bist100_df= generate_features_for_df(bist100_df_base,hasVolume=False)





        #Generate exchange features
        print("Generating exchange features")
        usdtry_df= generate_features_for_df(usdtry_df_base,has_volume=False, no_extra=True)
        brltry_df= generate_features_for_df(brltry_df_base,has_volume=False, no_extra=True)
        rubtry_df= generate_features_for_df(rubtry_df_base,has_volume=False, no_extra=True)
        jpytry_df= generate_features_for_df(jpytry_df_base,has_volume=False, no_extra=True)



  
    
    if use_all_features: 
        cols_to_remove_for_exchange = ["Exchange","High", "Low"]
        cols_to_remove_for_index = ["High", "Low"]


        #JOIN FEATURES into single dataframe
        stock_to_predict = stock_df #thyao_df akbnk_df, ttrak_df, tcell_df, hekts_df, alctl_df


        print("Base Stock shape:", stock_to_predict.shape)
        df2 = pd.merge(stock_to_predict,
                       usdtry_df.drop(cols_to_remove_for_exchange,axis=1)
                       ,how="inner", on="Date", suffixes=("_stock","_usd"))
        print("DF2 shape (stock+usd):", df2.shape)

        df2 = pd.merge(df2,
                       brltry_df.drop(cols_to_remove_for_exchange,axis=1)
                        ,how="inner", on="Date", suffixes=("_stock","_bry"))

        print("DF2 shape (df2+bry):", df2.shape)

        df2 = pd.merge(df2,
                       rubtry_df.drop(cols_to_remove_for_exchange,axis=1)
                        ,how="inner", on="Date", suffixes=("_bry","_rub"))
        print("DF2 shape (df2+rub):", df2.shape)

        df2 = pd.merge(df2,
                       jpytry_df.drop(cols_to_remove_for_exchange,axis=1)
                        ,how="inner", on="Date", suffixes=("_rub","_jpy"))
        print("DF2 shape (df2+jpy):", df2.shape)

        df2 = pd.merge(df2,
                       bist30_df.drop(cols_to_remove_for_index,axis=1)
                        ,how="inner", on="Date", suffixes=("_jpy","_bist30"))
        print("DF2 shape (df2+bist30):", df2.shape)

        ##Since google trend has different timestamp we want to have date column only
        df2['Date'] = df2['Date'].dt.normalize()


        #join google trend data
        df2 = pd.merge(df2,interest_over_time_df,how="inner", on="Date", suffixes=("_m","_goog"))
        print("DF2 shape (df2+google trend):", df2.shape)


    else:
        df2 = stock_df
    


    #Remove stocks raw data excep Close
    cols_to_remove_final = ["High", "Low","Stock","Volume"]
    ##Close_stock will be removed finaly, need date and close index from train/test data
    df2.drop(cols_to_remove_final,axis=1,inplace=True)

    print("DF2 shape before Nan drop:", df2.shape)

    #if still has nan drop colms
    #drop columns that has more than 20% nan
    df2 = df2.dropna(thresh=2/10*len(df2), axis=1)
    print("DF2 shape (drop 20% nan colmns):", df2.shape)


    #drop any row with nan ##likely first rows due to technical indicator calculations!
    df2.dropna(inplace=True)
    print("DF2 shape after dropping na:", df2.shape)


    df2= df2.sort_values(by='Date').reset_index(drop=True)

    ##CLEAN

    # Clean NaN values
    #df2 = df2.drop(["faiz"],axis=1)
    #df2 = df2.dropna()
    #df2 = clean_dataset(df2).reset_index(drop=True) #need to remove date for clean_dataset
    df2 = df2.reset_index(drop=True)
    df2= clean_dataset(df2)
    print("DF2 shape after cleanup:", df2.shape)



    ##copy df2 somewhere so index and date and close can be used
    orig_df2_before_train = df2.copy()
    #classification
    target_next_day = df2["Next_day_prediction"]
    #regression
    target_next_day_regression = df2["Next_day_buy_price"]
    print("original df2 copyied to orig_df2_before_train:")



    #usd_nex_day = df2["Next_day_prediction_y"]
    #bry_nex_day = df2["Next_day_prediction_a"]

    filter_next_day = [col for col in df2 if col.startswith('Next_day_')]
    filter_close = [col for col in df2 if col.startswith('Close')]
    ##Do not delete close columns for all
    #filter_close = []

    #Date column no more
    #df2.drop(["Date","Close", *filter_next_day],axis=1,inplace=True)
    df2.drop(["Date", *filter_next_day, *filter_close],axis=1,inplace=True)
    print("DF2 shape after final drop:", df2.shape)




    ##GET TEST TRAIN split

    X = df2
    y = target_next_day

    X_train, X_test, y_train, y_test = capstone_module.split_train_test(X,y,percent=train_percent)   

    train_len = round((train_percent/100) * len(X))



    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report,confusion_matrix
    from sklearn.metrics import fbeta_score

    def train_and_predict(clf, X_train, y_train, X_test, y_test , params=None):
        """ Traing classifier and make score on predictions"""
        print("Training classifier: {}".format(clf))
        clf = clf.fit(X_train, y_train)
        return predict_labels(clf,X_test,y_test) 



    def predict_labels(clf, X_test,y_test):
        print("Predicting Test")
        predictions= clf.predict(X_test)
        clf_rep = classification_report(y_test,predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        fbeta= fbeta_score(y_test, predictions, 0.5 ,average='weighted')
        acc_score = accuracy_score(y_test,predictions)
        print("Scores:")
        print(clf_rep)
        print("F1 Score: ", f1)
        print("F0.5 Score:", fbeta)
        print("Accuracy Score: ",acc_score)
        
        return (predictions, clf_rep,f1,acc_score)
        
      



    #OPTIMIZE SVC using Grid search
    #Note that we use time series split to cross validate

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report,confusion_matrix


    #svc grid search - all features 
    num_features = X_train.shape[1]
    param_grid = {'kernel':( 'rbf','sigmoid'), 
                  'C':[0.5,1,2,3,7], 
                  #'C':[0.5,1,3,5], 
                  #'gamma':[0.001,0.01,0.1,1/num_features, 0.2,0.5,0.9],
                  'gamma':[0.0015,0.015,0.15,0.1,0.5,1,1/num_features],
                  #'shrinking':[False,True],
                  #'decision_function_shape':['ovo','ovr']
                    }
                    

    # Initialize the classifier
    clf = SVC(random_state=31)


    # The scorers can be either be one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    scoring = {"recall","accuracy"}

    # this is a timeseries validation should be done as it is else false grid search result will be shown
    tscv = TimeSeriesSplit(n_splits=2)

    #perform grid search
    #Use accuracy and F1 as score and refit to better f1
    grid = GridSearchCV(clf,param_grid,verbose=0,cv=tscv,  scoring=scoring,refit="accuracy")
    # May take awhile!
    grid.fit(X_train,y_train)

    print("Best parameters and best estimator")
    clf = grid.best_estimator_
    print(grid.best_params_)
    print(grid.best_estimator_)

    grid_pred = predict_labels(clf,X_test,y_test)

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report,confusion_matrix
    from sklearn.metrics import fbeta_score

    clf_rep = classification_report(y_test,grid_pred[0])
    f1 = f1_score(y_test, grid_pred[0], average='binary')
    fbeta= fbeta_score(y_test, grid_pred[0], 0.5 ,average='binary')
    acc_score = accuracy_score(y_test,grid_pred[0])
    print("Scores:")
    print(clf_rep)
    print("F1 Score: ", f1)
    print("F0.5 Score:", fbeta)
    print("Accuracy Score: ",acc_score)


    ##SIMULATION

    prediction_to_use = np.array(grid_pred[0])


    ###TODO THIS ONLY PREDICTION, should also take that days price !!
    #Grid SVC simulation on test
    input_money = 100
    plt.figure()
    if use_all_features:
        close_col = "Close_stock"
    else:
        close_col = "Close"
    day_worth_df=simulate_outcome(prediction_to_use,
                                         orig_df2_before_train[train_len:][["Next_day_prediction","Date",close_col]],
                                         close_col=close_col,
                                         sell_all=True,buy_all=True,debug =False)


    import matplotlib
    colors = ['red','gray','green']
    plt.plot(day_worth_df["Date"],day_worth_df["Worth"])

    #plt.scatter(day_worth_df["Date"],day_worth_df["Worth"], c=day_worth_df["Action"], cmap=matplotlib.colors.ListedColormap(colors))

    plt.title("Stock:{} Prediction Trading vs Buy and Hold Benchmark".format(stock_name))
    ##BUY and hold
    input_money = 100
    day_worth_df=simulate_outcome(prediction_to_use,orig_df2_before_train[train_len:][["Next_day_prediction","Date",close_col]],close_col=close_col,sell_all=True,buy_all=True,debug =False, buy_and_hold=True)
    
    plt.plot(day_worth_df["Date"],day_worth_df["Worth"])
    
    plt.legend(['Predictive Trading', 'Buy and Hold'], loc='upper left')
    plt.ylabel("TRY")
    plt.xlabel("Days")
    plt.show()


