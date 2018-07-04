# Machine Learning Engineer Nanodegree
## Specializations
## Project: Predicting stock prices in Turkish Stock Market 
## Görkem Göknar 2018

**Note**

DISCLAIMER REGARDING FORECAST AND PREDICTIONS:
Financial Results and coding work in this project are done for educational purposes only. Actual stock performance may differ from those predicted by this work. No guarantee is presented or implied as accuracy of forecasts contained. I am not responsible for any financial loss made using this report and code provided.



Project results can be seen in provided jupyter notebook file: Udacity_Machine_Learning_Capstone.ipynb

Python 3.6 is used for code implementation.

Data used in the project is stored in data/ folder where each dataframe can be reused by reading from csv file. Example implementation is provided in jupyter notebook of the project.
data/akbnk_df_base.csv
data/rubtry_df_base.csv
data/bist30_df_base.csv
data/brltry_df_base.csv
data/interest_over_time_df.csv
data/usdtry_df_base.csv
data/jpytry_df_base.csv

Following custom libraries are needed for jupyter notebook provided by this project to function.

Capstone_module.py
* get_stock_from_mynet : Pulls stock data from mynet.com and returns as pandas dataframe
* get_exchange_from_mynet : Pulls exchange data from mynet.com and returns as pandas dataframe
* get_stock_features : transforms raw data to ratio data and adds extra date features
* generate_ta_features : generates technical analysis features using technical analysis library
* generate_features_for_df : combines get_stock_features and generate_ta_features with date normalization
* normalize_series : normalizes given series to [0,1] range
* clean_dataset : clears nan values and returns any null values
* split_train_test : splits data to training and test with given percentage
* simulate_outcome : simulates trading, using given close price  and predictions, outcome and trades are store in return. if monkey_trading = True, randomized guesses are made to simulate typing monkeys.


Evaluate_model.py
* predict_stock : Systemized training and testing of provided stock data. Given stock dataframe only, builds prediction system and does the simulation . If use_all_features=True additionally pulls bist30, usd/try, bry/try, rub/try and does the simulation accordingly. Must be run on Jupyter Notebook.

google_trend.py
* get_google_trend_data : Uses pytrends to pull google trend data with given keywords. Based on https://github.com/GeneralMills/pytrends/blob/master/pytrends/dailydata.py


Additional Public Libraries used:

Pandas - https://pandas.pydata.org
Numpy - http://www.numpy.org
pytrends  - https://github.com/GeneralMills/pytrends
sklearn   - http://scikit-learn.org
keras  - http://keras.io



Please email [machine-support@udacity.com](mailto:machine-support@udacity.com) if you have any questions.
