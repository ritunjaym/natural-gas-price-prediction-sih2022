#import dependencies
from turtle import color, down
import streamlit as st
import pandas as pd
import plotly.express as px
import itertools
import statsmodels.api as sm
import numpy as np
import time
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

padding_top = 0

# def update_plot(fig,date,start):
#     if start==True:
#         fig.update_layout(xaxis_range=[])



st.markdown(f"""
    <style>
        .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Sidebar Implementation for file upload

# st.sidebar.subheader("Please upload a CSV File")

st.sidebar.subheader("Please upload a CSV file in the below given format")
st.sidebar.image("./template.jpeg",clamp=True)

uploaded_file = st.sidebar.file_uploader(label ="",type=['csv'])


if 'checked' not in st.session_state:
    st.session_state['checked'] = False

if 'upload_count' not in st.session_state:
    st.session_state['upload_count'] = False


def updateDate(fig,date):
    fig.update

global df
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.to_csv('ngpp-data-daily.csv')
    except Exception as e:
        print(e)
# st.sidebar.subheader("Please upload a csv file in the below given format")
# st.sidebar.image("./template.jpeg",clamp=True)
    
def forecast():
    typeofchart = st.selectbox('Select the type of Visualization',options=['Forecasted Prices','Price on a specific Date'])
    st.session_state.checked=True
    if(typeofchart == 'Forecasted Prices'):
        
        df = pd.read_csv('Forecast_monthly.csv')
        # df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')
        df.style.format(precision=3) 
        fig = px.line(df, x='Date', y='Mean',title="Natural Gas Price Prediction from 2021-2026",markers=True,labels={"Date":"Year","Mean":"Price in US Dollars($)"})
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward",),
                    dict(count=1, label="1M", step="month", stepmode="backward",),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD",step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig.update_layout({
            'margin_b' : 0,
            'margin_l' : 0,
            'height' : 450,
        })
        fig.update_layout(template='plotly_dark',
                  xaxis_rangeselector_font_color='white',
                  xaxis_rangeselector_activecolor='black',
                  xaxis_rangeselector_bgcolor='black',
                 )

        col1, col2 = st.columns([1, 1])

        minDate=datetime.datetime.strptime('2021-01-01', "%Y-%m-%d").date()
        maxDate=datetime.datetime.strptime('2026-01-01', "%Y-%m-%d").date()

        with col1:
            start=str(st.date_input('Select the Start Date Interval',value=minDate,min_value=minDate,max_value=maxDate))
        with col2:
            end=str(st.date_input('Select the End Date Interval',value=maxDate,min_value=minDate,max_value=maxDate))
        df2 = pd.read_csv('Forecast.csv')
        d = pd.Series(df.Mean.values,index=df.Date).to_dict()
        
        s = float()
        e = float()

        if start in d.keys():
            s = d[start]
        if end in d.keys():
            e = d[end]
        
        percentage=((e-s)/s)*100
        # precentage=str(percentage)+'%'
        lowCol, highCol, = st.columns(2)
        lowCol.metric(f"Price on {start}", f'${s}', )
        highCol.metric(f"Price on {end}", f'${e}', f'{round(percentage,2)}%')
        


        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')

        

        fig.update_layout(xaxis=dict(range=[start,end]))
        st.plotly_chart(fig,use_container_width=True)
        
        df.drop('Unnamed: 0',axis=1,inplace=True)
        st.sidebar.download_button(label="Download Output CSV File",data=df2.to_csv().encode('utf-8'),file_name=f"Forecast {datetime.datetime. now(). strftime('%Y-%m-%d-%I:%M:%S_%p')}.csv",mime='text/csv')
    elif typeofchart=='Price on a specific Date':
        inpDate, dummy = st.columns([1, 1])
        with inpDate:
            selected_date = st.date_input('Please select the Date to get the price')
        selected_date =  selected_date.strftime("%Y-%m-%d")
        # selected_date=datetime.datetime.strptime(selected_date, '%Y/%m/%d').strftime('%d-%m-%Y')
        df=pd.read_csv('Forecast.csv')
        d = pd.Series(df.Mean.values,index=df.Date).to_dict()
        
        if selected_date in d.keys():
            col1, col2 = st.columns(2)
            col1.metric("Predicted Internaltional Natural Gas Price in US Dollars",f'$ {round(d[selected_date],2)}')
            col2.metric("Predicted International Natural Gas Price in INR", f'₹ {round(d[selected_date]*79.93,2)}')
        else:
            print(selected_date)
        st.sidebar.download_button(label="Download Output CSV File",data=df.to_csv().encode('utf-8'),file_name=f"Forecast {datetime.datetime. now(). strftime('%Y-%m-%d-%I:%M:%S_%p')}.csv",mime='text/csv')
        
def model():    
    df = pd.read_csv('upload_file.csv')
    df['Date']=pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')

    y = df['Spot_price'].resample('MS').mean()
    p = range(0, 2)
    d = range(0, 2)
    q = range(0, 2)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order = param, seasonal_order = param_seasonal, enforce_stationary = False,enforce_invertibility=False) 
                result = mod.fit()   
            except: 
                continue
    model = sm.tsa.statespace.SARIMAX(y, order = (1, 1, 1),
                                    seasonal_order = (1, 1, 0, 12)
                                    )
    result = model.fit(disp=0)
    prediction = result.get_prediction(start = pd.to_datetime('2019-01-01'), dynamic = False)
    prediction_ci = prediction.conf_int()
    Date = prediction_ci.index

    y_hat = prediction.predicted_mean
    y_truth = y['2019-01-01':]
    mse = ((y_hat - y_truth) ** 2).median()
    rmse = np.sqrt(mse)

    pred_uc = result.get_forecast(steps = 50)
    print(pred_uc)
    pred_ci = pred_uc.conf_int()

    cols=['lower Spot_price','upper Spot_price']
    values=pred_ci[cols].mean()
    pred_ci.insert(2,"Mean",values)
    pred_ci['Mean'] = pred_ci[['lower Spot_price', 'upper Spot_price']].mean(axis=1)
    pred_ci.reset_index(inplace=True)
    pred_ci.rename(columns = {'index':'Date'}, inplace = True)
    pred_ci.to_csv('Forecast_monthly_2.csv')
    

   


def stats_page():
    typeofchart = st.selectbox('Select the type of visualization',options=['Commodities','Bar Graph','Seasonal Plot'])
   
    if typeofchart == 'Commodities':
        df = pd.read_csv('multiple_commodities.csv')
        df.head()
        df.drop('Unnamed: 0',axis=1,inplace=True)
        fig1 = px.line(df, x='Date', y=df.columns.drop(labels=['Date']),title="Commodities Price Chart")#,fill="tonexty",color_discrete_sequence=["green", "orange", "black"]
        fig1.update_layout(showlegend=True)
        st.plotly_chart(fig1,use_container_width=True)

    elif typeofchart == 'Bar Graph':
        data = pd.read_csv('consumption_vis.csv')
        data = data.drop('Natural Gas Total Consumption',axis=1)
        s1 = data.sum()
        values = s1.values
        labels = s1.index
        fig = px.bar(x=labels,y=values,color=values)
        fig.update_traces(marker_color='#F1A661')
        fig.update_layout({
        'height':600,
        'xaxis_title':"Categories",
        'yaxis_title':"Value in Million Cubic Feet (MMCF)",
        })

        st.plotly_chart(fig,use_container_width=True)
    elif typeofchart == 'Seasonal Plot':
        data_orig = pd.read_csv('ngpp-data.csv')
        data_orig['Date'] = pd.to_datetime(data_orig['Date']) 
            
        #Deseasonalize the graph
        data_orig.set_index('Date', inplace=True)
        analysis = data_orig[['Spot_price']].copy()

        decompose_result_mult = seasonal_decompose(analysis, model="additive")
        trend = decompose_result_mult.trend
        seasonal = decompose_result_mult.seasonal
        residual = decompose_result_mult.resid

        plt = decompose_result_mult.plot()
        st.pyplot(plt)






if uploaded_file:
    st.header("Predicted Data Dashboard")
    with st.spinner('Model is running...'):
        if st.session_state.checked==False:
            st.session_state.upload_count+=1
            model()
            print("Model ran")
            

        forecast()
    
    
else:
    st.header("Historical Data Dashboard")
    stats_page()

