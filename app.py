import streamlit as st
import requests

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

##############################################################################

# Section 2:  Explaining the data of the project

##############################################################################

# Section 3:  Discuss the modelling and the results of modelling

##############################################################################

# Section 4:  Showcase how we would have liked the app to work

##############################################################################

# Section 5: Conclusions

##############################################################################
