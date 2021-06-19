
from __future__ import print_function
import streamlit as st
import requests
import pymysql
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime as dt
from plotly.subplots import make_subplots
import numpy as np
import time
from PIL import Image
import mplcyberpunk
from datetime import datetime
import time
import matplotlib.pyplot as plt
from pandas import datetime
from IPython import get_ipython
from concurrent.futures import ThreadPoolExecutor

option = st.sidebar.selectbox('Selecciona una opción',('Mercado de divisas', 'Criptomonedas', 'Bitcoin/Eurodolar', 'Modelos'))
st.header(option)


    
if option =='Mercado de divisas':
    st.subheader('Datos y gráfico Mercado de divisas')
    snp500 = pd.read_csv("divisas.csv")
    symbols = snp500['Symbol'].sort_values().tolist()
    

    ticker = st.sidebar.selectbox(
        'Choose a Stock',
         symbols)

    i = st.sidebar.selectbox(
            "Interval in minutes",
            ( "1m", "5m", "15m", "30m", "1h")
        )


    p = st.sidebar.number_input("How many days (1-30)", min_value=1, max_value=30, step=1)


    stock = yf.Ticker(ticker)
    history_data = stock.history(interval = i, period = str(p) + "d")

    prices = history_data['Close']
    volumes = history_data['Volume']

    lower = prices.min()
    upper = prices.max()
    prices_ax = np.linspace(lower,upper, num=20)

    vol_ax = np.zeros(20)

    for i in range(0, len(volumes)):
        if(prices[i] >= prices_ax[0] and prices[i] < prices_ax[1]):
            vol_ax[0] += volumes[i]   

        elif(prices[i] >= prices_ax[1] and prices[i] < prices_ax[2]):
            vol_ax[1] += volumes[i]  

        elif(prices[i] >= prices_ax[2] and prices[i] < prices_ax[3]):
            vol_ax[2] += volumes[i] 

        elif(prices[i] >= prices_ax[3] and prices[i] < prices_ax[4]):
            vol_ax[3] += volumes[i]  

        elif(prices[i] >= prices_ax[4] and prices[i] < prices_ax[5]):
            vol_ax[4] += volumes[i]  

        elif(prices[i] >= prices_ax[5] and prices[i] < prices_ax[6]):
            vol_ax[5] += volumes[i] 

        elif(prices[i] >= prices_ax[6] and prices[i] < prices_ax[7]):
            vol_ax[6] += volumes[i] 

        elif(prices[i] >= prices_ax[7] and prices[i] < prices_ax[8]):
            vol_ax[7] += volumes[i] 

        elif(prices[i] >= prices_ax[8] and prices[i] < prices_ax[9]):
            vol_ax[8] += volumes[i] 

        elif(prices[i] >= prices_ax[9] and prices[i] < prices_ax[10]):
            vol_ax[9] += volumes[i] 

        elif(prices[i] >= prices_ax[10] and prices[i] < prices_ax[11]):
            vol_ax[10] += volumes[i] 

        elif(prices[i] >= prices_ax[11] and prices[i] < prices_ax[12]):
            vol_ax[11] += volumes[i] 

        elif(prices[i] >= prices_ax[12] and prices[i] < prices_ax[13]):
            vol_ax[12] += volumes[i] 

        elif(prices[i] >= prices_ax[13] and prices[i] < prices_ax[14]):
            vol_ax[13] += volumes[i] 

        elif(prices[i] >= prices_ax[14] and prices[i] < prices_ax[15]):
            vol_ax[14] += volumes[i]   

        elif(prices[i] >= prices_ax[15] and prices[i] < prices_ax[16]):
            vol_ax[15] += volumes[i] 

        elif(prices[i] >= prices_ax[16] and prices[i] < prices_ax[17]):
            vol_ax[16] += volumes[i]         

        elif(prices[i] >= prices_ax[17] and prices[i] < prices_ax[18]):
            vol_ax[17] += volumes[i]         

        elif(prices[i] >= prices_ax[18] and prices[i] < prices_ax[19]):
            vol_ax[18] += volumes[i] 

        else:
            vol_ax[19] += volumes[i]

    fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.2, 0.8],
            specs=[[{}, {}]],
            horizontal_spacing = 0.01
        )

    fig.add_trace(
            go.Bar(
                    x = vol_ax, 
                    y= prices_ax,
                    text = np.around(prices_ax,2),
                    textposition='auto',
                    orientation = 'h'
                ),
            row = 1, col =1
        )

    dateStr = history_data.index.strftime("%d-%m-%Y %H:%M:%S")

    fig.add_trace(
        go.Candlestick(x=dateStr,
                    open=history_data['Open'],
                    high=history_data['High'],
                    low=history_data['Low'],
                    close=history_data['Close'],
                    yaxis= "y2"  
                ),
            row = 1, col=2
        )

    fig.update_layout(
        title_text='Market Profile Chart (FOREX)', # title of plot
        bargap=0.01, # gap between bars of adjacent location coordinates,
        showlegend=False,

        xaxis = dict(
                showticklabels = False
            ),
        yaxis = dict(
                showticklabels = False
            ),

        yaxis2 = dict(
                title = "Price",
                side="right"
            )
    )

    fig.update_yaxes(nticks=20)
    fig.update_yaxes(side="right")
    fig.update_layout(height=800)

    config={
            'modeBarButtonsToAdd': ['drawline']
        }

    st.plotly_chart(fig, use_container_width=True, config=config)
    
    
if option =='Criptomonedas':
    st.subheader('Datos y gráfico Criptomonedas')
    snp500 = pd.read_csv("crypto2.csv")
    symbols = snp500['Symbol'].sort_values().tolist()
    

    ticker = st.sidebar.selectbox(
        'Choose a Cryptocurrencie',
         symbols)

    i = st.sidebar.selectbox(
            "Interval in minutes",
            ( "15m", "30m", "1h")
        )


    p = st.sidebar.number_input("How many days (1-30)", min_value=1, max_value=30, step=1)


    stock = yf.Ticker(ticker)
    history_data = stock.history(interval = i, period = str(p) + "d")

    prices = history_data['Close']
    volumes = history_data['Volume']

    lower = prices.min()
    upper = prices.max()
    prices_ax = np.linspace(lower,upper, num=20)

    vol_ax = np.zeros(20)

    for i in range(0, len(volumes)):
        if(prices[i] >= prices_ax[0] and prices[i] < prices_ax[1]):
            vol_ax[0] += volumes[i]   

        elif(prices[i] >= prices_ax[1] and prices[i] < prices_ax[2]):
            vol_ax[1] += volumes[i]  

        elif(prices[i] >= prices_ax[2] and prices[i] < prices_ax[3]):
            vol_ax[2] += volumes[i] 

        elif(prices[i] >= prices_ax[3] and prices[i] < prices_ax[4]):
            vol_ax[3] += volumes[i]  

        elif(prices[i] >= prices_ax[4] and prices[i] < prices_ax[5]):
            vol_ax[4] += volumes[i]  

        elif(prices[i] >= prices_ax[5] and prices[i] < prices_ax[6]):
            vol_ax[5] += volumes[i] 

        elif(prices[i] >= prices_ax[6] and prices[i] < prices_ax[7]):
            vol_ax[6] += volumes[i] 

        elif(prices[i] >= prices_ax[7] and prices[i] < prices_ax[8]):
            vol_ax[7] += volumes[i] 

        elif(prices[i] >= prices_ax[8] and prices[i] < prices_ax[9]):
            vol_ax[8] += volumes[i] 

        elif(prices[i] >= prices_ax[9] and prices[i] < prices_ax[10]):
            vol_ax[9] += volumes[i] 

        elif(prices[i] >= prices_ax[10] and prices[i] < prices_ax[11]):
            vol_ax[10] += volumes[i] 

        elif(prices[i] >= prices_ax[11] and prices[i] < prices_ax[12]):
            vol_ax[11] += volumes[i] 

        elif(prices[i] >= prices_ax[12] and prices[i] < prices_ax[13]):
            vol_ax[12] += volumes[i] 

        elif(prices[i] >= prices_ax[13] and prices[i] < prices_ax[14]):
            vol_ax[13] += volumes[i] 

        elif(prices[i] >= prices_ax[14] and prices[i] < prices_ax[15]):
            vol_ax[14] += volumes[i]   

        elif(prices[i] >= prices_ax[15] and prices[i] < prices_ax[16]):
            vol_ax[15] += volumes[i] 

        elif(prices[i] >= prices_ax[16] and prices[i] < prices_ax[17]):
            vol_ax[16] += volumes[i]         

        elif(prices[i] >= prices_ax[17] and prices[i] < prices_ax[18]):
            vol_ax[17] += volumes[i]         

        elif(prices[i] >= prices_ax[18] and prices[i] < prices_ax[19]):
            vol_ax[18] += volumes[i] 

        else:
            vol_ax[19] += volumes[i]

    fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.2, 0.8],
            specs=[[{}, {}]],
            horizontal_spacing = 0.01
        )

    fig.add_trace(
            go.Bar(
                    x = vol_ax, 
                    y= prices_ax,
                    text = np.around(prices_ax,2),
                    textposition='auto',
                    orientation = 'h'
                ),
            row = 1, col =1
        )

    dateStr = history_data.index.strftime("%d-%m-%Y %H:%M:%S")

    fig.add_trace(
        go.Candlestick(x=dateStr,
                    open=history_data['Open'],
                    high=history_data['High'],
                    low=history_data['Low'],
                    close=history_data['Close'],
                    yaxis= "y2"  
                ),
            row = 1, col=2
        )

    fig.update_layout(
        title_text='Market Profile Chart (Cryptocurrencies)', # title of plot
        bargap=0.01, # gap between bars of adjacent location coordinates,
        showlegend=False,

        xaxis = dict(
                showticklabels = False
            ),
        yaxis = dict(
                showticklabels = False
            ),

        yaxis2 = dict(
                title = "Price (USD)",
                side="right"
            )
    )

    fig.update_yaxes(nticks=20)
    fig.update_yaxes(side="right")
    fig.update_layout(height=800)

    config={
            'modeBarButtonsToAdd': ['drawline']
        }

    st.plotly_chart(fig, use_container_width=True, config=config)
        
        
if option =='Bitcoin/Eurodolar':
    
    
    st.subheader('STOCK DASHBOARD')
    START = '2021-01-01'
    TODAY = dt.date.today().strftime('%Y-%m-%d')
    snp500 = pd.read_csv("btc_eur.csv")
    symbols = snp500['Symbol'].sort_values().tolist()
    ticker = st.sidebar.selectbox('Choose an option',symbols)
    i = st.sidebar.selectbox(
            "Interval in minutes",
            ( "15m", "30m", "1h")
        )


    p = st.sidebar.number_input("How many days (1-30)", min_value=1, max_value=30, step=1)
   
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    data_load_state = st.text('Load data...')
    data = load_data(ticker)
    data_load_state.text('loading data...done')
    st.subheader('Data')
    st.write(data.tail())
 

    stock = yf.Ticker(ticker)
    history_data = stock.history(interval = i, period = str(p) + "d")

    prices = history_data['Close']
    volumes = history_data['Volume']

    lower = prices.min()
    upper = prices.max()
    prices_ax = np.linspace(lower,upper, num=20)

    vol_ax = np.zeros(20)

    for i in range(0, len(volumes)):
        if(prices[i] >= prices_ax[0] and prices[i] < prices_ax[1]):
            vol_ax[0] += volumes[i]   

        elif(prices[i] >= prices_ax[1] and prices[i] < prices_ax[2]):
            vol_ax[1] += volumes[i]  

        elif(prices[i] >= prices_ax[2] and prices[i] < prices_ax[3]):
            vol_ax[2] += volumes[i] 

        elif(prices[i] >= prices_ax[3] and prices[i] < prices_ax[4]):
            vol_ax[3] += volumes[i]  

        elif(prices[i] >= prices_ax[4] and prices[i] < prices_ax[5]):
            vol_ax[4] += volumes[i]  

        elif(prices[i] >= prices_ax[5] and prices[i] < prices_ax[6]):
            vol_ax[5] += volumes[i] 

        elif(prices[i] >= prices_ax[6] and prices[i] < prices_ax[7]):
            vol_ax[6] += volumes[i] 

        elif(prices[i] >= prices_ax[7] and prices[i] < prices_ax[8]):
            vol_ax[7] += volumes[i] 

        elif(prices[i] >= prices_ax[8] and prices[i] < prices_ax[9]):
            vol_ax[8] += volumes[i] 

        elif(prices[i] >= prices_ax[9] and prices[i] < prices_ax[10]):
            vol_ax[9] += volumes[i] 

        elif(prices[i] >= prices_ax[10] and prices[i] < prices_ax[11]):
            vol_ax[10] += volumes[i] 

        elif(prices[i] >= prices_ax[11] and prices[i] < prices_ax[12]):
            vol_ax[11] += volumes[i] 

        elif(prices[i] >= prices_ax[12] and prices[i] < prices_ax[13]):
            vol_ax[12] += volumes[i] 

        elif(prices[i] >= prices_ax[13] and prices[i] < prices_ax[14]):
            vol_ax[13] += volumes[i] 

        elif(prices[i] >= prices_ax[14] and prices[i] < prices_ax[15]):
            vol_ax[14] += volumes[i]   

        elif(prices[i] >= prices_ax[15] and prices[i] < prices_ax[16]):
            vol_ax[15] += volumes[i] 

        elif(prices[i] >= prices_ax[16] and prices[i] < prices_ax[17]):
            vol_ax[16] += volumes[i]         

        elif(prices[i] >= prices_ax[17] and prices[i] < prices_ax[18]):
            vol_ax[17] += volumes[i]         

        elif(prices[i] >= prices_ax[18] and prices[i] < prices_ax[19]):
            vol_ax[18] += volumes[i] 

        else:
            vol_ax[19] += volumes[i]

    fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.2, 0.8],
            specs=[[{}, {}]],
            horizontal_spacing = 0.01
        )

    fig.add_trace(
            go.Bar(
                    x = vol_ax, 
                    y= prices_ax,
                    text = np.around(prices_ax,2),
                    textposition='auto',
                    orientation = 'h'
                ),
            row = 1, col =1
        )

    dateStr = history_data.index.strftime("%d-%m-%Y %H:%M:%S")

    fig.add_trace(
        go.Candlestick(x=dateStr,
                    open=history_data['Open'],
                    high=history_data['High'],
                    low=history_data['Low'],
                    close=history_data['Close'],
                    yaxis= "y2"  
                ),
            row = 1, col=2
        )

    fig.update_layout(
        title_text='Market Profile Chart ', # title of plot
        bargap=0.01, # gap between bars of adjacent location coordinates,
        showlegend=False,

        xaxis = dict(
                showticklabels = False
            ),
        yaxis = dict(
                showticklabels = False
            ),

        yaxis2 = dict(
                title = "Price ",
                side="right"
            )
    )

    fig.update_yaxes(nticks=20)
    fig.update_yaxes(side="right")
    fig.update_layout(height=800)

    config={
            'modeBarButtonsToAdd': ['drawline']
        }

    st.plotly_chart(fig, use_container_width=True, config=config)
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text='Time series data', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()
   
if option =='Modelos':
   
    st.subheader('Modelos Machine Learning')
    modelos= ('Arima', 'Comparación ARIMA - LSTM')
    selected_stock = st.selectbox('Select', modelos)
    
    if selected_stock =='Arima':
        df = pd.read_csv("dataset_BTC_USD.csv")
        train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
        predicciones_arima = pd.read_csv("arima_predictions_BTCUSD.csv")
  
        train_ar = train_data['Open'].values
        history = [x for x in train_ar]

        initial_date_train = train_data.index[0]
        end_date_train = train_data.index[-1]

        initial_date_test = test_data.index[0]
        end_date_test = test_data.index[-1]
    
        test_ar = test_data['Open'].values
    
        def grafica_train_test(df, test_data):
    
            fig = go.Figure()

    # Add traces

            plt.figure(figsize=(12,7))

            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Open'],
                                mode='lines',
                                name='Training Data'))
    
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=test_data['Open'],
                                mode='lines',
                                name='Test Data',
                                marker_color = 'black'))
    
            fig.update_layout(title='Conjuntos de datos empleado para el par Bitcoin - Dólar',
                              xaxis_title="Fecha (Mes Día Año)",
                              yaxis_title="Precio de apertura ($)")
            st.plotly_chart(fig)
        grafica_train_test(df, test_data)
        
        def compara_prediccion(df, test_data, predictions):
  

            fig = go.Figure()

    # Add traces

            plt.figure(figsize=(12,7))

            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Open'],
                                mode='lines',
                                name='Training Data'))
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=predictions['Predictions'],
                                mode='lines+markers',
                                name='Predicted Price', marker_color = 'red'))
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=test_data['Open'],
                                mode='lines+markers',
                                name='Test Data', marker_color = 'black'))
            fig.update_layout(title='Datos y predicciones para el par Bitcoin - Dólar',
                              xaxis_title="Fecha (HH:mm Mes Día, Año)",
                              yaxis_title="Precio de apertura ($)")


            st.plotly_chart(fig)
        compara_prediccion(df, test_data, predicciones_arima)
        
        def ARIMA_diff(df, train_data):

   ## Gráfico para diferenciación de grado 1:

            df_stocks_diff = train_data['Open']-train_data['Open'].shift()
    
            fig = go.Figure()

    # Add traces

            plt.figure(figsize=(12,7))

            fig.add_trace(go.Scatter(x=df['Datetime'], y=df_stocks_diff,
                                mode='lines',
                                name='Training Data'))
            fig.update_layout(title='Diferencia entre el precio actual y el precio anterior (d=1) para el par Bitcoin - Dólar',
                              xaxis_title="Fecha (Mes Día Año)",
                              yaxis_title="Diferencia entre el precio actual y el anterior ($)")#"$p_n-p_{(n-1)}(\$$")

            st.plotly_chart(fig)
        ARIMA_diff(df, train_data)
        
    if selected_stock =='Comparación ARIMA - LSTM':
        df = pd.read_csv("dataset_BTC_USD.csv")
        train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
        predicciones_arima = pd.read_csv("arima_predictions_BTCUSD.csv")
        predicciones_lstm = pd.read_csv("predictions_rmsprop_lstm.csv")
        train_ar = train_data['Open'].values
        history = [x for x in train_ar]

        initial_date_train = train_data.index[0]
        end_date_train = train_data.index[-1]

        initial_date_test = test_data.index[0]
        end_date_test = test_data.index[-1]
    
        test_ar = test_data['Open'].values
        def comparacion_de_modelos(test_data, predicciones_arima, predicciones_lstm):
    

            fig = go.Figure()

    # Add traces

            plt.figure(figsize=(12,7))

            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=test_data['Open'],
                                mode='lines',
                                name='Conjunto de testeo', marker_color = 'blue'))
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=predicciones_arima['Predictions'],
                                mode='lines+markers',
                                name='Precios predichos por ARIMA', marker_color = 'green'))
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=predicciones_lstm['Predictions'],
                                mode='lines+markers',
                                name='Precios predichos por LSTM', marker_color = 'black'))
            fig.update_layout(title='Comparación de las mejores predicciones de los modelos para el par Bitcoin - Dólar',
                              xaxis_title="Fecha (HH:mm Mes Día, Año)",
                              yaxis_title="Precio de apertura ($)")


            st.plotly_chart(fig)
        comparacion_de_modelos(test_data, predicciones_arima, predicciones_lstm)
        
        def todo_en_uno(df, test_data, predicciones_arima, predicciones_lstm):
   

            fig = go.Figure()

    # Add traces

            plt.figure(figsize=(12,7))
    
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Open'],
                                mode='lines',
                                name='Training Data', marker_color = 'grey'))
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=test_data['Open'],
                                mode='lines',
                                name='Conjunto de testeo', marker_color = 'blue'))
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=predicciones_arima['Predictions'],
                                mode='lines+markers',
                                name='Precios predichos por ARIMA', marker_color = 'green'))
            fig.add_trace(go.Scatter(x=test_data['Datetime'], y=predicciones_lstm['Predictions'],
                                mode='lines+markers',
                                name='Precios predichos por LSTM', marker_color = 'black'))
            fig.update_layout(title='Comparación de las mejores predicciones de los modelos para el par Bitcoin - Dólar',
                              xaxis_title="Fecha (HH:mm Mes Día, Año)",
                              yaxis_title="Precio de apertura ($)")


            st.plotly_chart(fig)
        todo_en_uno(df, test_data, predicciones_arima, predicciones_lstm)
   
  
