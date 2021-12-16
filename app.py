import streamlit as st
import pandas as pd
import numpy as np
import requests
from multipage import MultiPage
from pages import page1,page2  # import your pages here




#blabla
# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title("Global Food Price Predictor")

# Add all your applications (pages) here
app.add_page("Showcasing", page1.app)
app.add_page("Description", page2.app)


# The main app
app.run()


print(app)
