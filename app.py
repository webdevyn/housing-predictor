import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and the scaler
@st.cache_resource
def load_model():
    with open('models/linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# App title
st.title("House Price Prediction App")
st.write("Enter the details of the house to predict its price.")

# ------------------------------
# Sidebar inputs for main features
# ------------------------------
st.sidebar.header("House Features")

OverallQual  = st.sidebar.slider('Overall Quality', 1, 10, 6)
OverallCond  = st.sidebar.slider('Overall Condition', 1, 10, 5)
YearBuilt    = st.sidebar.slider('Year Built', 1872, 2010, 1970)
YearRemodAdd = st.sidebar.slider('Year Remodeled', 1950, 2010, 1990)
MasVnrArea   = st.sidebar.number_input('Masonry Veneer Area (sqft)', 0, 1600, 100)
GrLivArea    = st.sidebar.number_input('Above Ground Living Area (sqft)', 334, 5642, 1500)
LotArea      = st.sidebar.number_input('Lot Area (sqft)', 1300, 215000, 8000)
BedroomAbvGr = st.sidebar.slider('Bedrooms Above Ground', 0, 8, 3)
FullBath     = st.sidebar.slider('Full Bath', 0, 3, 2)

# ------------------------------
# Default values for remaining features
# ------------------------------
default_values = {
    'Id': 1000,
    'MSSubClass': 50,
    'LotFrontage': 70,
    'BsmtFinSF1': 400,
    'BsmtFinSF2': 0,
    'BsmtUnfSF': 500,
    'TotalBsmtSF': 900,
    '1stFlrSF': 1100,
    '2ndFlrSF': 500,
    'LowQualFinSF': 0,
    'BsmtFullBath': 1,
    'BsmtHalfBath': 0,
    'HalfBath': 0,
    'KitchenAbvGr': 1,
    'TotRmsAbvGrd': 6,
    'Fireplaces': 1,
    'GarageYrBlt': 1980,
    'GarageCars': 2,
    'GarageArea': 500,
    'WoodDeckSF': 100,
    'OpenPorchSF': 50,
    'EnclosedPorch': 0,
    '3SsnPorch': 0,
    'ScreenPorch': 0,
    'PoolArea': 0,
    'MiscVal': 0,
    'MoSold': 6,
    'YrSold': 2009
}

# ------------------------------
# Build features array in correct order
# ------------------------------
feature_order = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
                 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Fill in default values and overwrite with user inputs
features_dict = default_values.copy()
features_dict.update({
    'OverallQual': OverallQual,
    'OverallCond': OverallCond,
    'YearBuilt': YearBuilt,
    'YearRemodAdd': YearRemodAdd,
    'MasVnrArea': MasVnrArea,
    'GrLivArea': GrLivArea,
    'LotArea': LotArea,
    'BedroomAbvGr': BedroomAbvGr,
    'FullBath': FullBath
})

features = np.array([[features_dict[f] for f in feature_order]])

# ------------------------------
# Predict
# ------------------------------
if st.sidebar.button('Predict Price'):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    st.success(f'Estimated Price: ${prediction:,.2f}')
    
    st.subheader('Property Summary')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Overall Quality', OverallQual)
        st.metric('Overall Condition', OverallCond)
        st.metric('Year Built', YearBuilt)
        st.metric('Year Remodeled', YearRemodAdd)
        st.metric('Masonry Veneer Area (sqft)', MasVnrArea)
        st.metric('Above Ground Living Area (sqft)', GrLivArea)
        st.metric('Lot Area (sqft)', LotArea)
    
    with col2:
        st.metric('Bedrooms Above Ground', BedroomAbvGr)
        st.metric('Full Bath', FullBath)
# Info section
st.markdown('---')
st.subheader('About this Model')
st.write("""
This is a Linear Regression model trained on a dataset of housing data.
Features used for prediction include various numeric property attributes, such as:

- Overall Quality and Overall Condition
- Year Built and Year Remodeled
- Basement and Floor Areas (1st, 2nd, Total Basement, Finished/Unfinished)
- Living area above ground
- Number of bedrooms, bathrooms, kitchens, and total rooms above ground
- Fireplaces
- Garage details (year built, number of cars, area)
- Porch and deck areas (Wood Deck, Open Porch, Enclosed Porch)
- Other numeric features related to lot size, masonry, and miscellaneous property metrics
""")