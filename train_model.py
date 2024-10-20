import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the dataset
cars_data = pd.read_csv('Cardetails.csv')

# Function to extract numeric values from strings
def extract_numeric(value):
    try:
        return float(value.split(' ')[0].replace(',', ''))
    except (ValueError, AttributeError):
        return float('nan')

# Apply the function to numeric columns with potential string values
cars_data['mileage'] = cars_data['mileage'].apply(extract_numeric)
cars_data['engine'] = cars_data['engine'].apply(extract_numeric)
cars_data['max_power'] = cars_data['max_power'].apply(extract_numeric)
cars_data['torque'] = cars_data['torque'].apply(extract_numeric)

# Handle missing values by filling with the median
cars_data['mileage'] = cars_data['mileage'].fillna(cars_data['mileage'].median())
cars_data['engine'] = cars_data['engine'].fillna(cars_data['engine'].median())
cars_data['max_power'] = cars_data['max_power'].fillna(cars_data['max_power'].median())

# Replace categorical values with numerical labels
cars_data['owner'] = cars_data['owner'].replace(
    ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
    [0, 1, 2, 3, 4]
)

cars_data['fuel'] = cars_data['fuel'].replace(
    ['Diesel', 'Petrol', 'LPG', 'CNG', 'Electric'],
    [0, 1, 2, 3, 4]
)

cars_data['seller_type'] = cars_data['seller_type'].replace(
    ['Individual', 'Dealer', 'Trustmark Dealer'],
    [0, 1, 2]
)

cars_data['transmission'] = cars_data['transmission'].replace(
    ['Manual', 'Automatic'], [0, 1]
)

# Convert 'name' to category and then to numerical codes
cars_data['name'] = cars_data['name'].astype('category').cat.codes

# Define the feature set X and the target variable y
X = cars_data.drop(['selling_price'], axis=1)
y = cars_data['selling_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    enable_categorical=True
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Print performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained and saved as 'model.pkl'.")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plotting the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Prediction')
plt.title('Predicted vs Actual Selling Prices')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.legend()
plt.grid()
plt.show()
