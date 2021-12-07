import streamlit as st

st.markdown('# Food Prediction API')

st.markdown('## Predict the price of a food based on nutrition values ')

food = st.text_input('Predict the price of:', 'Carrots')
st.write('Displaying the price of', food)
