import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

# Load the model
model = pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction ML Model')

# Load car data
cars_data = pd.read_csv('Cardetails.csv')

# Function to get the brand name from the car name
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

# Apply function to extract car brand names
cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Streamlit input fields
name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)
torque = st.slider('Torque', 50, 300)  # Added Torque input

if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, torque]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats', 'torque']
    )
    
    # Replace categorical variables with numerical values
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
                                       [1, 2, 3, 4, 5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)

    # Encode the car name
    input_data_model['name'] = input_data_model['name'].astype('category').cat.codes

    # Ensure the order of the columns is correct
    expected_columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']
    input_data_model = input_data_model[expected_columns]

    # Print expected and actual feature names for debugging
    model_feature_names = model.get_booster().feature_names
    st.write("Model Feature Names: ", model_feature_names)
    st.write("Input Feature Names: ", input_data_model.columns.tolist())

    # Predict the car price
    car_price = model.predict(input_data_model)

    st.markdown(f'Car Price is going to be: ₹{car_price[0]:,.2f}')
