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

## Deployment to Streamlit Cloud

This project is ready to deploy on **Streamlit Community Cloud**. The following items are included:

- `app.py` as the main application file
- `requirements.txt` with all Python dependencies (including `streamlit`)
- `runtime.txt` specifying Python 3.11
- `.streamlit/config.toml` with server settings for cloud

To deploy:

1. Push the repository to a public GitHub repo (it already is).
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in.
3. Click **New app** and select the GitHub repo and branch (`main`).
4. Set the **Main file path** to `app.py` and click **Deploy**.

Streamlit will install packages from `requirements.txt` and use the runtime specified. You can make changes and redeploy by pushing new commits.

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
