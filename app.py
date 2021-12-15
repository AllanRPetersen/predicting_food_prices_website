from pandas.core.frame import DataFrame
import streamlit as st
import requests
import pandas as pd
import numpy as np
from tensorflow.keras import models
import joblib

nn_model = models.load_model('raw_data/neural_network_v1.h5')
preproc = joblib.load('raw_data/preprocessor_v1.joblib')

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

st.write('''
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
    ''')

st.write('''
    There are two main functions, one that look for a food type within "foundation" foods, and the other for branded foods.
    The third funtion uses the other two and has a fail safe system in order to retrieve the most information possible,
    that's because some foods are estrictly processed foods, so they will be a branded food instead a of a fundation one.

    Here it is an example of on of the functions and the information it retrieves:


    ''')

from utils import nutrients_redux, nutrients_redux_2, nutrients_super

food = st.text_input('** Put the name of the food you want to retrieve info:')

display = nutrients_super(food)

display

st.write('''
    This applyied to our unique values of food table retrieved us a table of type of food with all of the mentioned information.
    We had to drop the cholesterol column since it was missing more than 30% of the data
    ''')

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
    We built a small pipeline to preprocess the needed data. The pipeline included a Min Max scaler for numerical data and One Hot Encoder for categorical data.
    Once fitted and transformed, the data was rady for modelling.
    '''
)
st.write(
    '''

    We tried different models: Basic linear regressor, KNN regressor, Ridge regressor and a SVR.
    We saw not so very good results with neither of them, so after trying and modifying different parameters of the models we found our "best" performance was given by the KNN regresor.
    Unfortunately it was below 50% of the variance explained and vary greatly after each "X" and "y" split, we suspected the data wasn't enough after all the cleaning and selection.
    ''')

st.markdown('### Second dataset')

st.write('''
    After the bad results we decided to repeat the whole process but this time with a New Zeland food dataset, unfortunately we got simillar results.
    So we tought the features we processed and selected weren't enough to explain the price change among the food products.
    ''')

st.markdown('### A new aproach')

st.write('''
    After the dissapointing results of the regression models, we decided to change the aim of the project and fit better the data we already had, so we changed our target, instead of predicting the numeric price,
    we predict if a food product would be expensive or cheap using the nutrients and the other features previously presented.
    ''')

st.write('''
    We first performed a Logistic regression and a Support Vector Classification, with both of them we found much better results.

    We had more than 80% of the data variance explained, with these better results we proceed to the next step.
    '''
)


st.markdown('### Neural Network Model')

st.write(
    '''
    We built a Neural Network to see if we get better results due to the variety of features we selected. For this task we used Tensorflow Keras.
    '''
)
st.write('''
    The network consisted in;\n
        - An input dense layer of 71 dimensions with 256 neurons with a relu activation. \n
        - A hidden dense layer of 128 neurons with a relu activation.\n
        - A second hidden dense layer of 64 neurons and a relu activation.\n
        - A third hidden dense layer with 32 neurons and a relu activation.\n
        - An output dense layer with 1 neuron for the binary output, and a sigmoid activation.\n
        - It also has a compiler with a binary crossentropy loss, adam optimizer and accuracy as its metric.
    ''')
from PIL import Image

image = Image.open('./raw_data/Loss_and_ Accuracy.png')

st.write('''
    We obtained an average loss and accuracy of 0.255 and 0.902 respectively, which is considered a good score, more if we compared with the baseline probability of predicting correctly based only on the data distribution.\n
    The baseline is 0.781, our model has an accuracy of 0.902.
    ''')

st.image(image, caption='Graph showing the loss and accuracy of the model over170 iterations')
##############################################################################

#Section 4:  Showcase how we would have liked the app to work
st.markdown('# Showcasing how we would have liked the APP to work')

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
# # Section 4.1: New model
st.markdown('# Our APP')

st.write('''Although our model was not able to predict the price of food.
    We were able to predict whether the food would be expensive or not, so right bellow an there is a showcase of how the model can be integrated in a web application.''')

st.markdown('# Food Prediction APP')

st.markdown(
    '## Classify whether the food will be expensive or not based on nutrition values '
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

food_cat = pd.read_csv('raw_data/food_categories.csv')
#st.write(food_cat.head())

category = st.selectbox("Food Category", food_cat)

index_2019 = st.text_input('CPI index 2019:', '10')
#st.write('  ') CPI value of the country, imput country

# # y = cheap_food_2['expensive']
# # X = cheap_food_2[['pt_name','protein','fat','carb','sugar','sodium','calcium','kcal','category','2019']]


def features():
    list_of_inputs = {
        'pt_name': [value],
        'protein': [(float(protein) / float(portion_size)) * 1000],
        'fat': [(float(fat) / float(portion_size)) * 1000],
        'carb': [(float(carb) / float(portion_size)) * 1000],
        'sugar': [(float(sugar) / float(portion_size)) * 1000],
        'sodium': [(float(sodium) / float(portion_size)) * 1000],
        'calcium': [(float(calcium) / float(portion_size)) * 1000],
        'kcal': [(float(kcal) / float(portion_size)) * 1000],
        'category': [category],
        '2019': [float(index_2019)]
    }
    inputs = DataFrame.from_dict(list_of_inputs)
    return inputs
    #return list_of_inputs


if st.button('Predict'):
    inputs = features()
    #Scaling dataframe
    scaled_inputs = preproc.transform(inputs)
    #Prediction
    prediction = nn_model.predict(scaled_inputs)
    st.write(prediction[0][0])

    #print(prediction[0][0])
    if prediction[0][0] > 0.5:

        result ='Expensive'
        result

    else:
        result ='Not Expensive'
        result

else:
    st.write('I was not clicked 😞')

# api_key = 'HbaYhOLNuzBDKvfs0qvVEB4Ymu1PxQmru9YdXvv2jfc'
# query = food
# response = requests.get(
#     f'https://api.unsplash.com/search/photos?client_id={api_key}&query={query}'
# )
# image_small = response.json()['results'][0]['urls']['small']
# st.image(
#     image_small,
#     caption=
#     f' {response.json()["results"][0]["alt_description"]} by {response.json()["results"][0]["user"]["name"]}'

##############################################################################

# Section 5: Conclusions

st.markdown('## Deployment')

st.write('''
    For deployment first we had to make a docker container because the model was big enough to not be able to load into heroku.
    After that we deployed the webapp with heroku
    ''')
st.markdown('# Conclusions')



st.write(
    '''
    Since we tried different aproaches, we have various conclusions:
    '''
)
st.write(
    '''
    - Machine learning regression models did not suit properly the problem we tried to solve, at least not with the features we have at hand.
    - Our data was better suited for classification models.
    - Classification models such as logistic regression and support vector classification perfomed considerably better at making a prediction.
    - Neural Networks with a binary output layer worked better at classifying than regular machine learning methods, with aproximately 12% of better precision than baseline.
    '''
)

st.markdown('## Recommendations')
st.write(
    '''
    It is recommended to try the regression models with more data at a local or regional scale for price prediction in order to find out if that aim is achievable.
    '''
)
##############################################################################
