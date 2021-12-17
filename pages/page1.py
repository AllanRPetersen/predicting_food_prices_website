from pandas.core.frame import DataFrame
import streamlit as st
import requests
import pandas as pd
import numpy as np
#from tensorflow.keras import models
import joblib
from utils import nutrients_super, nutrients_redux, nutrients_redux_2
###############################################################


def app():
    st.markdown('## Predict if the food will be expensive or not')

    food = st.text_input('Enter the food you want to calculate:', 'Tacos')

    result_2 = nutrients_super(food)
    dict1 = {}
    for k, v in result_2.items():
        dict1[k] = [v]
    result_1 = pd.DataFrame.from_dict(dict1).replace('not found', np.nan)

    st.write(result_2)

    api_key = 'HbaYhOLNuzBDKvfs0qvVEB4Ymu1PxQmru9YdXvv2jfc'
    query = food
    response = requests.get(
        f'https://api.unsplash.com/search/photos?client_id={api_key}&query={query}'
    )
    image_small = response.json()['results'][0]['urls']['small']
    st.image(
        image_small,
        caption=
        f' {response.json()["results"][0]["alt_description"]} by {response.json()["results"][0]["user"]["name"]}'
    )

    cpi = pd.read_csv('production_data/country_cpi.csv')

    country = st.selectbox("Country:", cpi)

    if st.button('Predict'):
        answer = cpi.loc[cpi['adm0_name'] == country]['2019'].to_list()[0]
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
            'index_2019': answer
        }

        #st.write(answer)

        prediction = requests.get(
            'https://nnmodel-ynawzkr5xa-ew.a.run.app/predict/', params=params)

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
