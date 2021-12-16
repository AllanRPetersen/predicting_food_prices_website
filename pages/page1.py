from pandas.core.frame import DataFrame
import streamlit as st
import requests
import pandas as pd
import numpy as np
#from tensorflow.keras import models
import joblib
from utils import nutrients_super, nutrients_redux,nutrients_redux_2
###############################################################



def app():
    st.markdown('## Predict if the food will be expensive or not')



    food = st.text_input('put the food you want to calculate:')

    result_1 = nutrients_super(food)

    #result_1.replace('not found', np.nan, inplace=True)

    st.write(result_1)

    cpi = pd.read_csv('raw_data/country_cpi.csv')

    country = st.selectbox("Country", cpi)

    if st.button('Predict'):
        params = {
            'value': result_1['unit'],
            'portion_size': result_1['size'],
            'protein': result_1['protein'],
            'fat': result_1['fat'],
            'carb': result_1['carb'],
            'sugar': result_1['sugar'],
            'sodium': result_1['sodium'],
            'calcium': result_1['calcium'],
            'kcal': result_1['kcal'],
            'category': result_1['category'],
            'index_2019': 10
        }

        prediction = requests.get('https://nnmodel-ynawzkr5xa-ew.a.run.app/predict/', params=params)

        st.write('Probability: ', prediction.json())

        #print(prediction[0][0])
        if prediction.json() > 0.5:

            result = 'Expensive'
            st.write(result)

        else:
            result = 'Not Expensive'
            st.write(result)
    else:
        st.write('I was not clicked 😞')
    return 'hi 2'
