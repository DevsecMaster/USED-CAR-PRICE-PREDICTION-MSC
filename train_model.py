import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import seaborn as sns

# Load the dataset
cars_data = pd.read_csv('Cardetails.csv')
print(cars_data.columns)  # Verify the column names

# Function to extract numeric values from strings (remove non-numeric characters)
def extract_numeric(value):
    try:
        return float(value.split(' ')[0].replace(',', ''))  # Split to remove any unit and commas
    except (ValueError, AttributeError):
        return float('nan')  # Return NaN if conversion fails

# Apply the function to numeric columns with potential string values
for col in ['mileage', 'engine', 'power', 'torque']:  # Include 'torque' here
    if col in cars_data.columns:
        cars_data[col] = cars_data[col].apply(extract_numeric)

# Handle missing values for numeric columns
for col in ['mileage', 'engine', 'power', 'torque']:  # Include 'torque' here
    if col in cars_data.columns:
        cars_data[col] = cars_data[col].fillna(cars_data[col].median())

# Convert categorical columns to category or numerical encoding
categorical_columns = ['owner', 'fuel', 'seller_type', 'transmission', 'name', 'location']
for col in categorical_columns:
    if col in cars_data.columns:
        # Check for missing values in categorical columns and fill with a placeholder if needed
        cars_data[col] = cars_data[col].fillna('Unknown')
        cars_data[col] = cars_data[col].astype('category').cat.codes

# Drop irrelevant or duplicate columns
columns_to_drop = ['Unnamed: 0']  # Add any other unnecessary columns here
cars_data = cars_data.drop(columns=columns_to_drop, errors='ignore')

# Ensure the target column is clean and drop rows with NaN or infinity in the target
target_column = 'price'  # Adjust if the target column has a different name
cars_data = cars_data.replace([float('inf'), -float('inf')], float('nan'))
cars_data = cars_data.dropna(subset=[target_column])

# Define the feature set X and the target variable y
X = cars_data.drop([target_column], axis=1, errors='ignore')
y = cars_data[target_column]

# Check for NaN or infinity in the feature set and target
print("Checking for NaN or infinity in feature set:")
print(X.isna().sum())
print("\nChecking for NaN or infinity in target:")
print(y.isna().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model with categorical support enabled
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    enable_categorical=True  # Ensure this is enabled for categorical types
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

# Confusion Matrix Approach (Classifying Price into Low, Medium, High)
# Using KBinsDiscretizer to categorize prices into bins
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
y_train_binned = kbins.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_binned = kbins.transform(y_test.values.reshape(-1, 1)).flatten()

# Now we will train a classifier based on the predicted values
y_pred_binned = kbins.transform(y_pred.reshape(-1, 1)).flatten()

# Confusion Matrix for the classified price bins
conf_matrix = confusion_matrix(y_test_binned, y_pred_binned)
print("\nConfusion Matrix for Price Categories:")
print(conf_matrix)

# Display the classification report
print("\nClassification Report:")
print(classification_report(y_test_binned, y_pred_binned))



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

# Plotting the residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()

# Plotting the distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='green')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid()
plt.show()

