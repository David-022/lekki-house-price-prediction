import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib


# Load Dataset
df = pd.read_csv("filtered_lagos_rent.csv")

# Removing duplicates
df_dup = df.copy().drop_duplicates()
print(f"Shape without duplicates: {df_dup.shape}")

df_dup = df_dup.drop(['Serviced'], axis=1)
#df_dup = df_dup.drop(['Furnished'], axis=1)
df_dup = df_dup.drop(['Newly Built'], axis=1)



# Removing outliers in price (prices below or above the 10th and 90th percentile respectively)
low, high = df_dup["Price"].quantile([0.1,0.9])
mask = df_dup["Price"].between(low, high)
df_dup = df_dup[mask]
print(df_dup.shape)

# Extract Abuja records
lekki_df = df_dup[df_dup["City"] == "Lekki"].drop(columns="City")


# Function that takes towns with less than 20 records and categorises them as others
def categorize_as_other(state, state_df):
    # get the counts of each town

    state_count = state_df["Neighborhood"].value_counts()
    
    # get towns less than 20
    less_than_20_town = state_count[state_count < 20].index.to_list()
    dict_less_than_20_town = {town: "Others" for town in less_than_20_town}
    
    # replacing each town with Others
    state_df.loc[:,"Neighborhood"] = state_df["Neighborhood"].replace(dict_less_than_20_town)
    
    return state_df 

# define function for grid search

def grid_search_cv(X_train, X_test, y_train, y_test):
    # Fit the OrdinalEncoder on the entire training set first
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded = ordinal_encoder.fit_transform(X_train)
    X_test_encoded = ordinal_encoder.transform(X_test)  # Transform the test set using the same encoder
    
    models_dict = {
        "Ridge": {
            "params": {"ridge__alpha": [1e-5, 1e-2, 1e-1, 1, 5, 10]},
            "estimator": make_pipeline(Ridge(random_state=42))
        },
        "DecisionTreeRegressor": {
            "params": {"decisiontreeregressor__max_depth": [2, 4, 10, 20]},
            "estimator": make_pipeline(DecisionTreeRegressor(random_state=42))
        },
        "RandomForestRegressor": {
            "params": {"randomforestregressor__max_depth": [2, 4, 10, 50],
                       "randomforestregressor__n_estimators": [5, 20, 50, 100, 200]},
            "estimator": make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1))
        },
        "GradientBoostingRegressor": {
            "params": {"gradientboostingregressor__max_depth": [2, 4, 10, 50],
                       "gradientboostingregressor__alpha": [1e-10, 1e-2, 0.5, 0.9],
                       "gradientboostingregressor__n_estimators": [5, 20, 50, 100, 200]},
            "estimator": make_pipeline(GradientBoostingRegressor(random_state=42))
        }
    }

    model_eval_dict = {"model_name": [], "best_parameter": [], "train_mae": [], "test_mae": [], "r_square": []}
    
    # Loop through each model
    for each_model in models_dict:
        each_model_dict = models_dict[each_model]
        params = each_model_dict["params"]
        estimator = each_model_dict["estimator"]
        
        # Perform Grid Search
        model_lag = GridSearchCV(param_grid=params, estimator=estimator, cv=5, scoring="neg_mean_absolute_error")
        model_lag.fit(X_train_encoded, y_train)

        test_mae = abs(model_lag.best_score_).astype(int)
        train_mae = abs(mean_absolute_error(y_train, model_lag.predict(X_train_encoded))).astype(int)
        
        # Store model evaluations
        model_eval_dict["model_name"].append(each_model)
        model_eval_dict["best_parameter"].append(model_lag.best_params_)
        model_eval_dict["train_mae"].append(train_mae)
        model_eval_dict["test_mae"].append(test_mae)
        model_eval_dict["r_square"].append(model_lag.best_estimator_.score(X_test_encoded, y_test))

    # Convert results into a DataFrame
    model_eval_df = pd.DataFrame(model_eval_dict).set_index("model_name")
    
    return model_eval_df
        


#Categorizing towns with less than 20 records as Others
lekki_df = categorize_as_other("lekki", lekki_df)
print(lekki_df["Neighborhood"].unique())


# Splitting data into training and testing data
X_abj, y_abj = lekki_df.drop(columns="Price"), lekki_df["Price"]
X_train_abj, X_test_abj, y_train_abj, y_test_abj = train_test_split(X_abj, y_abj, test_size=0.2, random_state=42)



y_pred_base_abj = [y_train_abj.mean()] * len(y_train_abj)

print(f"Baseline R-square: {r2_score(y_train_abj, y_pred_base_abj)}")
print(f"Baseline Model MAE (training): N{mean_absolute_error(y_train_abj, y_pred_base_abj):,.2f}")
print(f"Baseline Model MAE (test): N{mean_absolute_error(y_test_abj, [y_train_abj.mean()] * len(y_test_abj)):,.2f}")


# Run the grid search CV
model_eval_df_abj = grid_search_cv(X_train_abj, X_test_abj, y_train_abj, y_test_abj)
print(model_eval_df_abj)


#Selecting parameters of the model with the least test mae: model that generalized well with unseen records
print(model_eval_df_abj[model_eval_df_abj.test_mae == model_eval_df_abj.test_mae.min()]["best_parameter"].iloc[0])

# fitting our model with the best parameters
final_model_abj = make_pipeline(OrdinalEncoder(), GradientBoostingRegressor(random_state=42, alpha=1e-10, max_depth=4, n_estimators=100))
final_model_abj.fit(X_train_abj, y_train_abj)

print(f"Ridge R-square: {r2_score(y_train_abj, final_model_abj.predict(X_train_abj)):,.2f}")
print(f"Ridge Model MAE (training): N{mean_absolute_error(y_train_abj, final_model_abj.predict(X_train_abj)):,.2f}")
print(f"Ridge Model MAE (test): N{mean_absolute_error(y_test_abj, final_model_abj.predict(X_test_abj)):,.2f}")



# Step 11: Save the model
joblib.dump(final_model_abj, 'lagos_house_price_model.pkl')
print("Model saved successfully.")
