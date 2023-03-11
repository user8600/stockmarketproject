from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
from keras.models import load_model
from pandas_datareader import data as pdr
import streamlit as st
import streamlit.components.v1 as com


st.set_page_config(
    page_title="StockX"
)  


st.sidebar.title('Navigation')


def prediction():

    end = dt.date.today()
    start = end - dt.timedelta(days=3650)

    st.title('Stock Trend Prediction')

    stocks = pd.read_csv('Output 2.csv')
    stocks_list = stocks['Ticker'].values.tolist()

    yf.pdr_override()
    user_input = st.selectbox("Enter Stock Ticker", stocks_list, 0)
    df = pdr.DataReader(user_input, start, end)
    df = df.reset_index()
################################################################## Describing Data ############################################################
    st.subheader(str(user_input) + ' from ' + str(start) + ' to ' + str(end))
    st.write(df.describe())

################################################################### Data Insertion #########################################################
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    df.insert(3, "Moving average 100", ma100)
    df.insert(4, "Moving average 200", ma200)
################################################################### Visulalizations #########################################################


    st.subheader('Closing Price Vs Time Chart with 100 Moving Average')
    fig = px.line(df, x=df.Date, y=['Close', 'Moving average 100'])
    st.write(fig)

    st.subheader('Closing Price Vs Time Chart with 200 Moving Average')
    fig2 = px.line(df, x=df.Date,y=['Close', 'Moving average 100', 'Moving average 200'])
    st.write(fig2)


######################################################### Data Spliting and Scaling ########################################################

    data_training = pd.DataFrame(df['Close'][0:int(0.8*len(df))])
    data_testing = pd.DataFrame(df['Close'][int(0.8*len(df)):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling

    data_training_array = scaler.fit_transform(data_training)  # fitting

    model = load_model('sales_model.h5')  # Loding the Model


############################################################### Testing #####################################################################
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

############################################################## prdeiction ###############################################################
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    y_predicted = y_predicted.reshape(-1)
############################################################ Final Graph For Prediction ##########################################################
    temp_data = list(zip(df.Date[y_test.shape[0]: df.shape[0]], y_test, y_predicted))
    pre = pd.DataFrame(temp_data, columns=['Date', 'Testing Data', 'Prediction'])

    st.subheader('Prediction V/s Original Price')
    fig3 = px.line(pre, x=pre.Date, y=['Testing Data', 'Prediction'])
    st.write(fig3)
###################################################################################################################################################
    tickerSymbol = user_input
    tickerData = yf.Ticker(tickerSymbol)
    todayData = tickerData.history(period='1d')
    st.subheader('The Closing price for '+str(end)+' :')
    st.subheader(str(todayData['Close'][0]))



    

def home():
    col1,col2,col3=st.columns([1,2,1])
    col1.markdown("# StockX")
    col1.markdown("When it comes to investing, nothing will pay off more than educating yourself.Do the necessary research and analysis before making any investment decisions.")


def recom():
    stocks = pd.read_csv('Output 2.csv')
    stocks_list = stocks['Ticker'].values.tolist()

    yf.pdr_override()
    user_input = st.selectbox("Enter Stock Ticker", stocks_list, 0)
    u_input = yf.Ticker(user_input)

    info = u_input.get_info()
    col1,col2=st.columns([10,1])  
    col1.markdown(info)



options = st.sidebar.radio('Pages',options=['Home','Prediction','Information'])
st.sidebar.success('Click the above options')

if options=='Home':
    home()
elif options=='Prediction':
    prediction()
elif options=='Information':
    recom()
