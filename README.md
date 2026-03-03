# Housing Price Predictor

A machine learning project that predicts house prices based on property features.

## Project Goal

Build a regression model to predict housing prices using features like bedrooms, bathrooms, square footage, and year built.

## Dataset

- Source: CSV file downloaded from Kaggle
- Features (numeric used for prediction):
  - OverallQual, OverallCond, YearBuilt, YearRemodAdd, MasVnrArea
  - BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF
  - 1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea
  - BsmtFullBath, BsmtHalfBath, FullBath, HalfBath
  - BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, Fireplaces
  - GarageYrBlt, GarageCars, GarageArea
  - WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch
  - PoolArea, MiscVal, LotFrontage, LotArea, MSSubClass
  - MoSold, YrSold
- Target: SalePrice

## Technologies Used

- Python 3.8+
- Pandas, NumPy (data manipulation)
- Scikit-learn (machine learning)
- Matplotlib, Seaborn (visualization)
- Streamlit (web app)

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order (01, 02, 03)
4. Launch web app: `streamlit run app.py`

## Results

- Model: Linear Regression
- R² Score: 0.693 (Test set)
- RMSE: $35,293 (Test set)
- MAE: $25,764 (Test set)

## Future Improvements

- Try polynomial features
- Add regularization (Ridge/Lasso)
- Incorporate location data
- Deploy to cloud

## Author

Devyn Weir - Software Developer
