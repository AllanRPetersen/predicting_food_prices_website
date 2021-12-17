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
    st.markdown('## Predict whether the food will be expensive or not')

    st.write(
        '''
        This webapp evaluates if a given food is, or will be expensive or not based on it's nutrient information, type of food and country.
        The expensive or not expensive result represent a classification based on whether the cost of the food will be bellow or above 1 USD per kilogram.
        The model has been trained on  2020 prices from 71 countries around the world.
        '''
    )
    st.write(
        '''
        Once all the information is collected, we call our own API that uses cloud processing to perform the prediction.
        '''
    )
    ##################################################################
    'Radio function'

    display = st.radio('Select input mode', ('Automated', 'Manual'))

    if display == 'Automated':
        st.write(
            '''
            Automated mode uses a series of functions to look for information of the food into the USDA database and automatically retrieves it's nutrients, enery content and category.
            The CPI index is extracted for your selected country.
            '''
        )

        food = st.text_input('put the food you want to calculate:')

        result_2 = nutrients_super(food)
        dict1 = {}
        for k,v in result_2.items():
            dict1[k] = [v]
        result_1 = pd.DataFrame.from_dict(dict1).replace('not found', np.nan)

        st.write(result_2)

        cpi = pd.read_csv('raw_data/country_cpi.csv')

        country = st.selectbox("Country", cpi)

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
    if display == 'Manual':

        st.write(
            '''
            Manual mode skips the USDA API funtionality in order to make the calculations with your prefered values.
            Recommended in case you want to look for a strange or not registered food.
            '''
        )
        options = ("Retail", "Wholesale")

        value = st.selectbox("Retail or Wholesale", options)

        portion_size = st.text_input('Portion Size (g):', '2')
        st.write('  ')

        protein = st.text_input('Protein content :', '2')
        st.write('  ')

        fat = st.text_input('Fat content:', '3')
        st.write('  ')


        carb = st.text_input('Carbohydrate content:', '4')
        st.write('  ')

        sugar = st.text_input('Sugar content:', '5')
        st.write('  ')

        sodium = st.text_input('Sodium content:', '6')
        st.write('  ')

        calcium = st.text_input('Calcium content:', '7')
        st.write('  ')


        kcal = st.text_input('Amount of calories (kcal):', '8')
        st.write('  ')


        food_cat = pd.read_csv('production_data/food_categories.csv')
        #st.write(food_cat.head())

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


        category = st.selectbox("Food Category", food_cat)

        index_2019 = st.text_input('CPI index 2019:', '10')


        if st.button('Predict'):
            params = {
                'value': value,
                'portion_size': portion_size,
                'protein': protein,
                'fat': fat,
                'carb': carb,
                'sugar': sugar,
                'sodium': sodium,
                'calcium': calcium,
                'kcal': kcal,
                'category': category,
                'index_2019': index_2019
            }
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
