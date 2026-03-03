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

#app title
st.title("House Price Prediction App")
st.write("Enter the details of the house to predict its price.")

# Input fields for the features
st.sidebar.header("House Features")

# Create input fields
st.sidebar.header('House Features')

OverallQual     = st.sidebar.slider('Overall Quality', 1, 10, 6)
OverallCond     = st.sidebar.slider('Overall Condition', 1, 10, 5)
YearBuilt       = st.sidebar.slider('Year Built', 1872, 2010, 1970)
YearRemodAdd    = st.sidebar.slider('Year Remodeled', 1950, 2010, 1990)
MasVnrArea      = st.sidebar.number_input('Masonry Veneer Area (sqft)', 0, 1600, 100)
BsmtFinSF1      = st.sidebar.number_input('Basement Finished SF 1', 0, 5644, 400)
BsmtFinSF2      = st.sidebar.number_input('Basement Finished SF 2', 0, 1474, 0)
BsmtUnfSF       = st.sidebar.number_input('Basement Unfinished SF', 0, 2336, 500)
TotalBsmtSF     = st.sidebar.number_input('Total Basement SF', 0, 6110, 900)
FirstFlrSF      = st.sidebar.number_input('1st Floor SF', 334, 4692, 1100)
SecondFlrSF     = st.sidebar.number_input('2nd Floor SF', 0, 2065, 500)
GrLivArea       = st.sidebar.number_input('Above Ground Living Area (sqft)', 334, 5642, 1500)
BsmtFullBath    = st.sidebar.slider('Basement Full Bath', 0, 3, 1)
BsmtHalfBath    = st.sidebar.slider('Basement Half Bath', 0, 2, 0)
FullBath        = st.sidebar.slider('Full Bath', 0, 3, 2)
HalfBath        = st.sidebar.slider('Half Bath', 0, 2, 0)
BedroomAbvGr    = st.sidebar.slider('Bedrooms Above Ground', 0, 8, 3)
KitchenAbvGr    = st.sidebar.slider('Kitchens Above Ground', 0, 3, 1)
TotRmsAbvGrd    = st.sidebar.slider('Total Rooms Above Ground', 2, 14, 6)
Fireplaces      = st.sidebar.slider('Fireplaces', 0, 3, 1)
GarageYrBlt     = st.sidebar.slider('Garage Year Built', 1900, 2010, 1980)
GarageCars      = st.sidebar.slider('Garage Capacity (Cars)', 0, 4, 2)
GarageArea      = st.sidebar.number_input('Garage Area (sqft)', 0, 1418, 500)
WoodDeckSF      = st.sidebar.number_input('Wood Deck SF', 0, 857, 100)
OpenPorchSF     = st.sidebar.number_input('Open Porch SF', 0, 547, 50)
EnclosedPorch   = st.sidebar.number_input('Enclosed Porch SF', 0, 474, 0)
BsmtFinSF2      = st.sidebar.number_input('Basement Finished SF 2', 0, 1474, 0)  # Optional duplicate fix


# Collect all Streamlit inputs into a single features array
features = np.array([[
    OverallQual,
    OverallCond,
    YearBuilt,
    YearRemodAdd,
    MasVnrArea,
    BsmtFinSF1,
    BsmtFinSF2,
    BsmtUnfSF,
    TotalBsmtSF,
    FirstFlrSF,
    SecondFlrSF,
    GrLivArea,
    BsmtFullBath,
    BsmtHalfBath,
    FullBath,
    HalfBath,
    BedroomAbvGr,
    KitchenAbvGr,
    TotRmsAbvGrd,
    Fireplaces,
    GarageYrBlt,
    GarageCars,
    GarageArea,
    WoodDeckSF,
    OpenPorchSF,
    EnclosedPorch
]])

# Predict button
if st.sidebar.button('Predict Price'):
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Display result
    st.success(f'Estimated Price: ${prediction:,.2f}')
    
    # Show feature summary
    st.subheader('Property Summary')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric('Overall Quality', OverallQual)
        st.metric('Overall Condition', OverallCond)
        st.metric('Year Built', YearBuilt)
        st.metric('Year Remodeled', YearRemodAdd)
        st.metric('Masonry Veneer Area (sqft)', MasVnrArea)
        st.metric('Basement Finished SF 1', BsmtFinSF1)
        st.metric('Basement Finished SF 2', BsmtFinSF2)
        st.metric('Basement Unfinished SF', BsmtUnfSF)
        st.metric('Total Basement SF', TotalBsmtSF)
        st.metric('1st Floor SF', FirstFlrSF)
        st.metric('2nd Floor SF', SecondFlrSF)
    
    with col2:
        st.metric('Above Ground Living Area (sqft)', GrLivArea)
        st.metric('Basement Full Bath', BsmtFullBath)
        st.metric('Basement Half Bath', BsmtHalfBath)
        st.metric('Full Bath', FullBath)
        st.metric('Half Bath', HalfBath)
        st.metric('Bedrooms Above Ground', BedroomAbvGr)
        st.metric('Kitchens Above Ground', KitchenAbvGr)
        st.metric('Total Rooms Above Ground', TotRmsAbvGrd)
        st.metric('Fireplaces', Fireplaces)
        st.metric('Garage Year Built', GarageYrBlt)
        st.metric('Garage Capacity (Cars)', GarageCars)
        st.metric('Garage Area (sqft)', GarageArea)
        st.metric('Wood Deck SF', WoodDeckSF)
        st.metric('Open Porch SF', OpenPorchSF)
        st.metric('Enclosed Porch SF', EnclosedPorch)

# Info section
st.markdown('---')
st.subheader('About this Model')
st.write("""
This is a Linear Regression model trained on Calgary housing data.
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