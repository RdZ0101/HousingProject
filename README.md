This project provides solutions for predicting both rental and sales prices using historical data. Users can select various models for making predictions, such as Linear Regression, Random Forest, Support Vector Regression (SVR), and K-Nearest Neighbors (KNN).

The main.py script focuses on rental price predictions and its performance metrics. The Sales predictions is handled by the UserInput.py file that uses trained models saved in .pkl format. The model training is carried out in PredictionAnalysis.py file and finally Prediction.py file provides insights to performance metrics of the models that are being used for Sales Prediction.

The main.py script will ask the user whether they want prediction for Sales or Rent. If Sales is selected, it will call a function from UserInput.py, else it will stay in main.py and carry out the Rental price prediction along with providing insights.

Prerequisites Ensure the following Python libraries are installed:

pandas numpy scikit-learn matplotlib seaborn joblib

You can install them via: pip install pandas numpy scikit-learn matplotlib seaborn joblib

File Structure HousingProject { Prediction { Prediction.py PredictionAnalysis.py Preprocess.py UserInput.py Visualization.py main.py: Asks for Sales or Rent prediction. If Sales is selected, calls function in a different script. Script for rental price predictions based on user inputs like postcode, number of rooms, housing type, and prediction period. Preprocessor.py: Handles data preprocessing such as data cleaning, normalization, and outlier detection. Prediction.py: Contains models for sales price prediction using techniques like Random Forest, Linear Regression, and KNN. Datasets: CSV files for both rental and sales data, dynamically selected based on the user input. }

Prerequisits
User Input: Collects postcode, number of rooms, housing type (house, flat, unit), and months ahead for prediction.

Model Selection: Users choose from three machine learning models:


Predicted rent price Model performance metrics (MSE, MAE, R², RMSE) Bias-adjusted rent price 2. UserInput.py (Sales Price Prediction) The UserInput.py script is dedicated to sales price prediction based on trained models from PredictionAnalysis.py using the following models:

Random Forest Linear Regression K-Nearest Neighbors (KNN) It uses a different dataset (Melbourne_housing_FULL.csv) for sales data and evaluates models using cross-validation. Users are prompted to input Postcode, Number of rooms, House type and Landsize. A prediction will be given based on Random Forest model. For metrics it is requested to run the Prediction.py file.

Example Flow: Run PredictionAnalysis.py to train models and save them. Run main.py User input 1 for Sales prediction. Then user input Postcode, Number of rooms, House type and Landsize. A prediction will be given based on Random Forest model. For metrics it is requested to run the Prediction.py file.
Usage Running main.py (Rental Prediction) Follow the prompts to:

Select Sale or Rental options. Input the required details (postcode, rooms, housing type, landsize(optional)). Choose a model for rent prediction.

Running Prediction.py (Sales Prediction) You will be prompted to select a model for sales price prediction, and the script will output the evaluation metrics for the chosen model.

Sample Input for main.py:

Enter postcode, number of rooms, Type of housing(h=1,u=2,t=3,f=4), prediction period (months) separated by commas: 3000, 2, 4, 12 Sample Output for Rental Prediction: Predicted Rent Price for postcode 3000 in 12 months: $2000 using Linear Regression Adjusted Rent Price: $2100 Linear regression MSE: 5000, R²: 0.85, MAE: 200, RMSE: 70.71 Sample Output for Sales Prediction: Model: random_forest MAE: 30000 RMSE: 50000 R^2: 0.90 Datasets Rent_1BF_Final.csv: Rental data for 1-bedroom flats. Rent_2BF_Final.csv: Rental data for 2-bedroom flats. Rent_3BF_Final.csv: Rental data for 3-bedroom flats. Rent_2BH_Final.csv: Rental data for 2-bedroom houses. Rent_3BH_Final.csv: Rental data for 3-bedroom houses. Rent_4BH_Final.csv: Rental data for 4-bedroom houses. Melbourne_housing_FULL.csv: Sales data for Melbourne housing.
