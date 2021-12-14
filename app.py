import streamlit as st
import requests
import pandas as pd
import numpy as np

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
    '''We decide to use the name of the food to search for the nutritional information with the USDA API.
            To get a list of foods to search for we split the string in column cm_name (commitidy name)
            using a lambda function, generating a new column named type.
    ''')


price_df['type'] = price_df['cm_name'].apply(lambda x: x.split('-')[0])
st.write(price_df.head(3))

price_2020_df = price_df[price_df['mp_year'] == 2020]

st.write('''
    This new column was cleaned from any special charachcters (ex. !@#$$%*(-_/\)). Once cleaned,
            we make a new dataframe with the unique values (types of food) present in the dataset, so the
            API will look for the same information repeatedly, this is particularly important because the
            USDA API has a limit of 3,600 requests per hour.
    ''')



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

st.markdown('## API function construction')

st.write(
    '''
    In order to extract the information we wanted (nutrient values and food category) we couldn't just use the
    USDA raw API, because the response had many other parameters, so we had to built a series of functions
    that worked with the API.

    First of all, we selected a fixed list of nutrient information we were interested on (
        protein,
        fat,
        carbohydrates,
        sugar,
        sodium,
        calcium
        and cholesterol).Besides that, we also extracted the energy content in kcal, the portion size all of these nutrients were measured with,
        and the category of food it belongs according to the USDA data.
    '''
)

st.write(
    '''
    There are two main functions, one that look for a food type within "foundation" foods, and the other for branded foods.
    The third funtion uses the other two and has a fail safe system in order to retrieve the most information possible,
    that's because some foods are estrictly processed foods, so they will be a branded food instead a of a fundation one.

    Here it is an example of on of the functions and the information it retrieves:


    '''
)

from utils import nutrients_redux, nutrients_redux_2, nutrients_super

food = st.text_input('** Put the name of the food you want to retrieve info:')

display = nutrients_super(food)

display

st.write(
    '''
    This applyied to our unique values of food table retrieved us a table of type of food with all of the mentioned information.
    We had to drop the cholesterol column since it was missing more than 30% of the data
    '''
)


st.markdown('## Currency')

st.write('''The food price was given in the local currency.
         In total there were 62 currencies which we needed to convert to a single currency
         in order to compare the prices. The list of currencies were:  ''')

st.write(price_2020_df['cur_name'].unique())

st.write('''Each individual currency has a three letter abbreviation.
         The non-official abbreviation NIS (new Israeli shekel) was changed to ILS (Israeli new shekel).
         We passed a list of the abbreviations to an API called currencylayer.
         SSP (South Sudan Pound) had to be removed from the list as our API did not have a value for it.
         SPP was later added manually. The API returned us a list of conversion rates to US dollars as a JSON.
         This allowed us to convert the prices to US dollars.''')


def get_dataframe_data_2():

    return pd.read_csv('raw_data/price_with_USD.csv')


price_with_USD = get_dataframe_data_2()

st.write('Our dataframe with prices converted to US dollars:')
st.write(price_with_USD.head(3))

list_of_non_food = [
    'Charcoal ', 'Corn Soy Blend (CSB++, food aid) ', 'Cotton ',
    'Dishwashing liquid ', 'Disinfecting solution ', 'Fuel (Super Petrol) ',
    'Fuel (diesel) ', 'Fuel (diesel, parallel market) ', 'Fuel (kerosene) ',
    'Fuel (petrol', 'Handwash soap ', 'Laundry detergent ', 'Laundry soap ',
    'Salt ', 'Salt (iodised) ', 'Shampoo '
]
st.write(
    '''Looking over the unique values in the column named type, we discovered
         that a number of non-food items still remained in our data. These were: '''
)
st.write(list_of_non_food)

st.write(''' We used a for-loop to remove the non-food entries from our data.
         Afterwards the following columns were dropped: mp_year, mkt_name, adm1_name, mp_month, cur_name.'''
         )
st.write(
    '''The aforementioned columns do not add any information that could be used in our model to predict the food price.'''
)

st.write(
    '''The data was grouped by: the country (adm0_name), food type (type), retail or wholesale (pt_name), unit (um_name).
         The mean average was taken of: the original food price (mp_price), conversion rate (conv_rate) and food price in US dollars (usd_rate).
         This gave us a dataset that looked like this:''')


def get_dataframe_data_3():

    return pd.read_csv('raw_data/grouped_cleaned_data.csv')


grouped_cleaned_data = get_dataframe_data_3()

st.write(grouped_cleaned_data.head(3))

st.markdown('### Table join')
st.write('''
    After all the processing of the USDA nutrient api function and the currency conversion, we inner joined the tables, and further cleaned the new integrated table.
    ''')
##############################################################################

# Section 3:  Discuss the modelling and the results of modelling

st.markdown('## Modelling and results')

st.write(
    '''
    We tried different models: Basic linear regressor, KNN regressor, Ridge regressor and a SVR.
    We saw not so very good results with neither of them, so after trying and modifying different parameters of the models we found our "best" performance was given by the KNN regresor.
    Unfortunately it was below 50% of the variance explained and vary greatly after each "X" and "y" split, we suspected the data wasn't enough after all the cleaning and selection.
    '''
)

st.markdown('### Second dataset')

st.write(
    '''
    After the bad results we decided to repeat the whole process but this time with a New Zeland food dataset, unfortunately we got simillar results.
    So we tought the features we processed and selected weren't enough to explain the price change among the food products.
    '''
)

st.markdown('### A new aproach')

st.write(
    '''
    After the dissapointing results of the regression models, we decided to change the aim of the project and fit better the data we already had, so we changed our target, instead of predicting the numeric price,
    we predict if a food product would be expensive or cheap using the nutrients and the other features previously presented.
    '''
)

st.write(
    '''
    We first performed a Logistic regression and a Support Vector Classification, with both of them we found much better results.
    We had more than 80% of the data variance explained
    '''
)

##############################################################################

# Section 4:  Showcase how we would have liked the app to work
st.markdown('# Our APP')

st.write(
    '''Despite the limited success of our models ability predict the price of food,
          we will now showcase how our app should have worked. Below you can type in name of the food
          you wish to know the price of and the app will return a price and a picture of the food.'''
)

st.markdown('# Food Prediction APP')

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

# Section 5: Conclusions

##############################################################################
