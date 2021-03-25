# import library
import streamlit as st
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import joblib

@st.chache

def load(dataset):
    data = pd.read_csv(dataset)
    return data

image = Image.open('rain_image.jpg')
Context = """The objective about this project is to predict wheter tomorrow willbe rain or not default based on several parameters. Because I am not an expert in meteorology and climatology domain, to prevent bias in choosing the threshold, I will make the result of this app from binary classification to percentage of posibilty default. With this simple app, the banker or guarantor will be more efficient to predict which application should be accepted or declined.
The dataset is from the U.S. Small Business Administration (SBA) The U.S. SBA was founded in 1953 on the principle of promoting and assisting small enterprises in the U.S. credit market (SBA Overview and History, US Small Business Administration (2015)). Small businesses have been a primary source of job creation in the United States; therefore, fostering small business formation and growth has social benefits by creating job opportunities and reducing unemployment. 
There have been many success stories of start-ups receiving SBA loan guarantees such as FedEx and Apple Computer. However, there have also been stories of small businesses or start-ups that have defaulted on their SBA guaranteed loans. \n Small business owners often seek out SBA (Small Business Association) loans because they guarantee part of the loan. Without going into too much detail, this basically means that the SBA will cover some of the losses should the business default on the loan, which lowers the risk involved for the business owner(s). This increases the risk to the SBA however, which can sometimes make it difficult to get accepted for one of their loan programs.
This project is end to end data science projcet (kinda), from Data Preparation, Modeling, Evaluating, Tunning until Deployment. If you want to see more detail about this project, click this link below:"""

Link = 'https://github.com/farrasalyafi/sba_loan_default_prediction'

Linkedin = 'https://www.linkedin.com/in/muhammad-farras/'

# Main APP
def main():
    """Simple Rain Forecast"""  
    st.title('Rain Forecast App')
    st.write('By : Tubagus Moh Reja')

    # Menu
    menu = ['Prediction', 'About this App']
    choiche = st.sidebar.selectbox('Select Activities', menu)

    if choiche == 'About this App':
        st.header('About this Project')
        st.image(image, width=570)
        st.write(Context)
        st.write(Link)
        st.write(Linkedin)
    if choiche == 'Prediction':
        st.image(image, width=600)
        # Making Widget input
        st.write('Observation on Parameters at 9 am')
        Humidity9am = st.slider('Percentage of Humidity at 9 am Today ? (%)', 0, 100)
        Pressure9am = st.slider('How much Atmospheric pressure at 9 am ? (hpa)', 980, 1050)
        Cloud9am = st.slider('How much Fraction of Sky obscured by cloud at 9 am? (oktas)',0,8)
        st.write('Observation on Parameters at 3 pm')
        Humidity3pm = st.slider('How much Percentage of Humidity at 3 pm Today ? (%)', 0,100)
        Pressure3pm = st.slider('How much Atmospheric pressure at 3 pm ? (hpa)', 980, 1050)
        Cloud3pm = st.slider('How much Fraction of Sky obscured by cloud at 3 pm? (oktas)',0,8)
        st.write('Observation on other Parameters')
        WindSpeed = st.slider('The Speed of strongest wind in the 24 hours ? (km/h)'0,100)
        Sunshine = st.slider('The number of hours of bright sunshine in the day ?', 0,24)
        Rainfall = st.slider('The ammount of rainfall recorder in the last 24 hours ? (mm)', 0,200)
        MinTemp = st.slider('The minimum temperature recorded in the last 24 hours', -10,40)

        # Compile the data for Forecasting
        input_data = [Humidity9am, Pressure9am, Cloud9am, Humidity3pm, Pressure3pm, Cloud3pm, WindSpeed, Sunshine, Rainfall, MinTemp]
        input_data = np.array(input_data).reshape(1,-1)

        # Forecasting
        if st.button('Forecast!'):
            forecaster = pickle.load(open('rain_forecast.pkl', 'rb'))
            forecasting = forecaster.predict(input_data)
            forecast_proba = forecaster.predict_proba(input_data)[:,1]
            proba_result = (str((np.around(float(forecast_proba),3)*100)) + '%')
            def get_result(forecasting):
                if forecasting == 0:
                    print('Not Rain')
                else:
                    print('Will be Rain')
            st.subheader('Prediction for Tomorrow is :' + get_result(forecasting))
            st.subheader('The Probability for that event happen is :' + proba_result)

if __name__ == '__main__':
    main()
