import streamlit as st
import requests
import pandas as pd
import numpy as np

st.markdown('# Food Prediction API')

st.markdown('## Predict the price of a food based on nutrition values ')

food = st.text_input('Predict the price of:', 'Carrots')
st.write('Displaying the price of', food)

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

##############################################################################

# Section 1:  Explaining the hypothesis of the project

#Hypothesis:
st.markdown('# Project Aim')
st.write(
    '''The aim of this project is to predict the price of food based on their nutrients.
    Our hypothesis was that food containing more nutrients would be more desirable,
    leading to increased demand and as a result being more expensive.
    In contrast foods having few nutrients would have a lower price.''')

##############################################################################

# Section 2:  Explaining the data of the project
st.markdown('# Data')
st.write(
    '''Price of the foods were found as a dataset called Global Food Prices Database on Kaggle.
This dataset contained over 1 million rows of food prices from different markets around the world.'''
)
st.write('The raw dataset is displayed below:')


#@st.cache
def get_dataframe_data():

    return pd.read_csv(
        'raw_data/Global Food Prices Database/wfp_food_prices_database.csv')


price_df = get_dataframe_data()

st.write(price_df.head())
st.write(f'Shape of dataset: {price_df.shape}')

st.markdown('## Data Cleaning')

st.write(
    '''We decide to use the name of the food to search for the nutritional information in the API (api name?).
            To get a list of foods to search for we split the string in column cm_name (commitidy name)
            using a lamda function, generating a new column: type ''')

price_df['type'] = price_df['cm_name'].apply(lambda x: x.split('-')[0])
st.write(price_df.head(3))

price_2020_df = price_df[price_df['mp_year'] == 2020]

st.write('''We aim to train our with the price of food in 2020.
         Hence, the dataframe was filtered by the year 2020 using Boolean logic.'''
         )

st.write(
    '''The dataset did not have that many missing values but required extensive cleaning before it could be used for modelling.
    The last column (mp_commoditysource) was empty and as dropped. We no longer needed cm_name.
    The column with the market name (adm1_name) contained 30% empty values. But was kept for now.
    All the id columns did not provide any useful information for our model so they were dropped.
    ''')
st.write(
    '''For the majority of the foods, the prices were given per KG or per L.
         To avoid lengthy conversions we filtered the data to only include these entries.'''
)

price_2020_df.drop(columns=[
    'adm0_id', 'adm1_id', 'mkt_id', 'cm_id', 'cur_id', 'cm_name', 'pt_id',
    'mp_commoditysource', 'um_id'
],
                   inplace=True)

price_2020_df = price_2020_df.loc[price_2020_df['um_name'].isin(['KG', 'L'])]

st.write('We now had a dataframe that looked like this:')

st.write(price_2020_df.head(3))

st.markdown('## Currency')

st.write('''The food price was given in the local currency.
         In total there were 62 currencies which we needed to convert to a single currency
         in order to compare the prices. The list of currencies were:  ''')

st.write(price_2020_df['cur_name'].unique())

st.write('''Each individual currency has a three letter abbreviation.
         We passed a list of the abbreviations to an API called currencylayer
         and it returned us a list of conversion rates to US dollars as a JSON.
         This allowed us to convert the prices to US dollars.''')

##############################################################################

# Section 3:  Discuss the modelling and the results of modelling

##############################################################################

# Section 4:  Showcase how we would have liked the app to work

##############################################################################

# Section 5: Conclusions

##############################################################################
