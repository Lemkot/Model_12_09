# -*- coding: utf-8 -*-
"""
API finance
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
#import numpy as np
import json5
from fastapi.logger import logger
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# 2. Create the app object
app = FastAPI()

@app.get('/')
async def price():
    try:        
        
        # Create a Yahoo Finance ticker objects
        stock_SP = yf.Ticker('^SPX')
        # stock_10y_futures = yf.Ticker('ZNZ23.CBT')
        # stock_3m_interest = yf.Ticker('^IRX')
        stock_10y_interest = yf.Ticker('^TNX')
        stock_vix_index = yf.Ticker('^VIX')
        
        # Fetch historical data for the stocks
        historical_data_SP = stock_SP.history(period='1y')
        # historical_data_10y_futures = stock_10y_futures.history(period='1y')
        # historical_data_3m_interest = stock_3m_interest.history(period='1y')
        historical_data_10y_interest = stock_10y_interest.history(period='1y')
        historical_data_vix_index = stock_vix_index.history(period='1y')
        
        # Extract the most recent closing prices
        prices_SP = historical_data_SP['Close']
        # prices_10y_futures = historical_data_10y_futures['Close']
        # prices_3m_interest = historical_data_3m_interest['Close']
        prices_10y_interest = historical_data_10y_interest['Close']
        prices_vix_index = historical_data_vix_index['Close']
        
        # Fit an ARIMA model to the training data
        order = (5, 1, 0)  # Example order for ARIMA (p, d, q)
        
        model_SP = ARIMA(prices_SP, order=order)
        model_fit_SP = model_SP.fit()
        
        # model_10y_futures = ARIMA(prices_10y_futures, order=order)
        # model_fit_10y_futures = model_10y_futures.fit()
        
        # model_3m_interest = ARIMA(prices_3m_interest, order=order)
        # model_fit_3m_interest = model_3m_interest.fit()
        
        model_10y_interest = ARIMA(prices_10y_interest, order=order)
        model_fit_10y_interest = model_10y_interest.fit()
        
        model_vix_index = ARIMA(prices_vix_index, order=order)
        model_fit_vix_index = model_vix_index.fit()
        
        # Forecast the next day's prices
        forecast_SP = model_fit_SP.forecast(steps=1).iloc[0]
        # forecast_10y_futures = model_fit_10y_futures.forecast(steps=1).iloc[0]
        # forecast_3m_interest = model_fit_3m_interest.forecast(steps=1).iloc[0]
        forecast_10y_interest = model_fit_10y_interest.forecast(steps=1).iloc[0]
        forecast_vix_index = model_fit_vix_index.forecast(steps=1).iloc[0]
        
        return {"forecasted_SP_index": float(forecast_SP), "forecasted_10y_interest": float(forecast_10y_interest), "forecasted_vix_index": float(forecast_vix_index)}

    except Exception as e:
        return {"error": str(e)}

    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
