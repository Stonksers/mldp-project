import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.title('Weather Classification')
st.write('Please enter the following data to predict the weather (Rainy, Cloudy, Sunny, or Snowy):')

temperature = st.number_input('Temperature (Celsius)')

uv_index = st.number_input('UV Index', min_value=0)

visibility = st.number_input('Visibility (km)')

humidity = 0.0
humidity_col1, humidity_col2 = st.columns((2, 1))

with humidity_col1:
    humidity_slider = st.slider('Humidity (%)', min_value=0.0, max_value=100.0, value=humidity, key='humidity_slider')
with humidity_col2:
    humidity_input = st.number_input('Or enter Humidity (%) here', min_value=0.0, max_value=100.0, value=humidity_slider)

humidity = humidity_input

precipitation = 0.0
precipitation_col1, precipitation_col2 = st.columns((2, 1))

with precipitation_col1:
    precipitation_slider = st.slider('Precipitation (%)', min_value=0.0, max_value=100.0, value=precipitation, key='precipitation_slider')
with precipitation_col2:
    precipitation_input = st.number_input('Or enter Precipitation (%) here', min_value=0.0, max_value=100.0, value=precipitation_slider)

precipitation = precipitation_input
precipitation

cloud_cover = st.selectbox('Cloud Cover', ('Clear', 'Partly Cloudy', 'Cloudy', 'Overcast'))

submit_button = st.button('Predict Weather', type='primary')

if submit_button:
    input_df = pd.DataFrame({
        'Temperature': [temperature], 
        'Humidity': [humidity], 
        'Precipitation (%)': [precipitation],
        'UV Index': [uv_index], 
        'Visibility (km)': [visibility],
        'Cloud Cover': [cloud_cover.lower()],
    })

    input_df = pd.get_dummies(input_df, columns=['Cloud Cover'])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    pred = model.predict(input_df)
    st.success(f'The predicted weather is: {pred[0]}')


