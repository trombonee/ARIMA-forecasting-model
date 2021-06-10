from QuantConnect.Data.Custom.Benzinga import *

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import ar_select_order

import numpy as np

import nltk
from vaderSentiment import SentimentIntensityAnalyzer, pairwise
from datetime import datetime, timedelta


class SimpleTimeSeries(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2019, 1, 1)  # Set Start Date
        self.SetCash(100000000)  # Set Strategy Cash
        
        self.equities = ["AAL","OSTK","BLNK","HRL","GME","DAL","AC"]
        self.newsSymbols = dict()
        
        #Adding equities we want to trade
        
        for equity in self.equities:
            self.AddEquity(equity, Resolution.Daily)
            self.newsSymbols[equity] = self.AddData(BenzingaNews, equity).Symbol
        
        self.Schedule.On(self.DateRules.EveryDay("AAL"), self.TimeRules.BeforeMarketClose("AAL", 30), self.rebalance)
        
        #Number of positions want to trade
        self.NUM_POSITIONS = 3
        
        #download sentiment data
        self.vaderData = self.Download("https://www.dropbox.com/s/q5udnl4ou35o78f/vader_lexicon.txt?dl=1")
        
    def rebalance(self):
        predictionRatio = dict()
        predictedPrices = dict()
        
        for key in self.equities:
            close_data = self.History([key], 45, Resolution.Daily)['close'].values
        
            #Set difference to 1 initially
            d = 0
            
            #Compute the p_value of the data (check for stationarity)
            p_value = adfuller(close_data)[1]
            
            #If the data is not stationary, then compute differences from 1-12
            if p_value > 0.05:
                
                #data is non-stationary and we require differencing
                close_data_diff = np.diff(close_data, 1)
                close_data_diff = close_data_diff[~np.isnan(close_data_diff)]
                diff_p_value = adfuller(close_data_diff)[1]
                    
                #If data is stationary move forward with the ARIMA, if not, try a 2nd difference. 
                if  diff_p_value < 0.05:
                    d = 1
                    close_data = close_data_diff
                else:
                    close_data_diff_2 = np.diff(close_data, 2)
                    close_data_diff_2 = close_data_diff_2[~np.isnan(close_data_diff_2)]
                    diff_p_value_2 = adfuller(close_data_diff_2)[1]
                    
                    if  diff_p_value_2 < 0.05:
                        d = 2
                        close_data = close_data_diff_2
                    else:
                        #If the data is non-stationary we skip this asset. 
                        continue
                    
            p_mod = ar_select_order(close_data, 6)
            if len( p_mod.ar_lags ) != 0:
                p = p_mod.ar_lags[-1]
            else:
                p = 1
            
            try:
                model = ARIMA(close_data, order=(p, d, 0))
                model_fit = model.fit(disp=0)
            except:
                model = ARIMA(close_data, order=(1, d, 0))
                model_fit = model.fit(disp=0)
                
            output = model_fit.forecast()
            prediction = output[0]
            
            currentPrice = close_data[-1]
            
            predictedPrices[key] = prediction
            predictionRatio[key] = prediction/currentPrice
        
        maxDict = predictionRatio.copy()
        
        #Get self.NUM_POSITIONS stocks with largest predicted return
        maxKeys = []
        for i in range(self.NUM_POSITIONS):
            temp = max(maxDict, key=maxDict.get)
            maxKeys.append(temp)
            maxDict.pop(temp)
        
        #Clear positions that have low predicted return
        for key in predictionRatio.keys():
            if key not in maxKeys and self.Securities[key].Invested:
                self.SetHoldings(key, 0)
                self.Debug(str(self.Time) + "Clearing: " + str(key) + " - Prediction: " + str(predictedPrices[key]))
        
        #Dictionaries for storing sentiment data
        weights = dict()
        numArticles = dict()
        average = dict()
        
        #For all the different stocks in our selection
        for key in maxKeys:
            
            #Sets initial values for the sentiment scores to zero
            weights[key] = 0.0  #Dict used to store individual sentiment scores
            numArticles[key] = 0.0  #Dict used to score number of articles for each stock
            average[key] = 0.0  #Dict used to hold average of the sentiment scores
            
            articles = []
            
            #Get news data for previous two days for all the selected stocks
            try:
                articles = self.History(BenzingaNews, self.newsSymbols[key], 1, Resolution.Daily)['contents'].values
            except:
                self.Debug(f"Key: {key} news not found")
            
            #Get sentiment for each article found
            for article in articles:
                text = article.lower()  #Gets string of the contents of the selected article
                   
                sentiment = SentimentIntensityAnalyzer(lexicon_file=self.vaderData) #Intilizes sentiment
                positivity = sentiment.polarity_scores(text)["compound"]    #Performs sentiment on entire article contents and returns compounnd score
                
                weights[key] += positivity  #Adds positivity to currrent weight to get sum
                numArticles[key] += 1.0     #Increases number of articles by 1
            
        #Average out sentiments sentiment data
        for stock in weights.keys():
            #If no articles make average zero
            if numArticles[stock] == 0.0:
                average[stock] = 0.0
            else:
                #Divide to find average
                average[stock] = float(weights[stock]) / float(numArticles[stock])  
                
        #Calculate available holdings
        availableHoldings = (1-(self.Portfolio.TotalHoldingsValue/self.Portfolio.TotalPortfolioValue))/self.NUM_POSITIONS
        
        if availableHoldings < 0:
            availableHoldings = 0
        
        #Invest in maxKeys positions with predicted return > 0.02% and account for sentiment analysis
        for max_key in maxKeys:
            if predictionRatio[max_key] > (1.002 - average[max_key]*0.005) and not self.Securities[max_key].Invested:
                self.SetHoldings(max_key, availableHoldings)
                self.Debug(str(self.Time) + " Purchasing " + str(max_key) + " - Prediction: " + str(predictedPrices[max_key]) + " Sentiment: " + str(average[str(max_key)]))
            elif predictionRatio[max_key] < (0.995 - average[max_key]*0.005) and self.Securities[max_key].Invested:
                self.SetHoldings(max_key, 0)
                self.Debug(str(self.Time) + " Selling " + str(max_key) + " - Prediction: " + str(predictedPrices[max_key]) + " Sentiment: " + str(average[str(max_key)]))
            else:
                self.Debug(f"Holding: {max_key} - Prediction: {predictedPrices[max_key]} Sentiment: {average[max_key]}")
        
    def OnData(self, data):
        return
