import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.title('Weather Classification')

st.text('')

st.subheader('Weather in the clouds, rain, sun, or snow, we always deliver the answers', divider=True)

st.text('Please enter the following data to classify the weather (Rainy, Cloudy, Sunny, or Snowy):')

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

cloud_cover = st.selectbox('Cloud Cover', ('Clear', 'Partly Cloudy', 'Cloudy', 'Overcast'))

st.text('')

col1, col2, col3 = st.columns((1, 1, 1))

with col2:
    submit_button = st.button('Classify Weather', type='primary', use_container_width=True)

st.text('')

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

    pred = model.predict(input_df)[0]
    st.success(f'The weather is classified as: {pred}')

    match pred:
        case 'Rainy':
            st.image('images/glass-window-1845534_1280.jpg')
        case 'Cloudy':
            st.image('images/clouds-4258726_1280.jpg')
        case 'Sunny':
            st.image('images/death-valley-3133502_1280.jpg')
        case 'Snowy':
            st.image('images/nature-7000445_1280.jpg')
        case _:
            st.text('Invalid Result')