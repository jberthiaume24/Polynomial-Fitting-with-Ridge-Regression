import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Load the data
D_x, D_y = np.loadtxt("train.dat", usecols=(0,1), unpack=True)
testData = np.loadtxt("test.dat", usecols=(0,1), unpack=True)

# Initialize storage for results
trainRMSEVals = [[] for _ in range(13)]
testRMSEVals = [[] for _ in range(13)]
best_poly_degree = None

# KFold cross-validation to select the best polynomial degree
kf = KFold(n_splits=6, random_state=None, shuffle=False)
for trainIndex, testIndex in kf.split(D_x):
    # Split data into training and testing sets
    x_train, x_test = D_x[trainIndex], D_x[testIndex]
    y_train, y_test = D_y[trainIndex], D_y[testIndex]

    # Initialize the scaler
    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()

    # Fit and transform training data
    x_train_scaled = scalerX.fit_transform(x_train.reshape(-1, 1))
    y_train_scaled = scalerY.fit_transform(y_train.reshape(-1, 1))
    
    # Transform testing data using the same scaler fitted on training data
    x_test_scaled = scalerX.transform(x_test.reshape(-1, 1))
    y_test_scaled = scalerY.transform(y_test.reshape(-1, 1))
    
    for d in range(13):  # Degree d ranges from 0 to 12
        # Apply polynomial feature transformation
        poly = PolynomialFeatures(degree=d)
        x_train_poly = poly.fit_transform(x_train_scaled)
        x_test_poly = poly.transform(x_test_scaled)

        # Create and train the model pipeline
        pipeline = make_pipeline(Ridge(alpha=0))
        pipeline.fit(x_train_poly, y_train_scaled)
        
        # Predict and calculate RMSE
        y_train_pred = pipeline.predict(x_train_poly)
        y_test_pred = pipeline.predict(x_test_poly)
        
        train_rmse = np.sqrt(mean_squared_error(scalerY.inverse_transform(y_train_scaled), scalerY.inverse_transform(y_train_pred)))
        test_rmse = np.sqrt(mean_squared_error(scalerY.inverse_transform(y_test_scaled), scalerY.inverse_transform(y_test_pred)))
        
        trainRMSEVals[d].append(train_rmse)
        testRMSEVals[d].append(test_rmse)

# Calculate average errors and find the best polynomial degree
avgTrainErr = [np.mean(trainRMSEVals[d]) for d in range(13)]
avgTestErr = [np.mean(testRMSEVals[d]) for d in range(13)]
best_poly_degree = np.argmin(avgTestErr)

# KFold Results
print("\n--- KFold Cross-Validation Results ---\n")
results_df = pd.DataFrame({
    'Degree': range(13),
    'Train RMSE': avgTrainErr,
    'Test RMSE': avgTestErr
})

# Print the results
print(results_df.to_string(index=False, formatters={'Train RMSE': '{:.4f}'.format, 'Test RMSE': '{:.4f}'.format}))

print(f"\nBest Polynomial Degree (from KFold): {best_poly_degree}")

# Final model training with the best polynomial degree
poly = PolynomialFeatures(degree=best_poly_degree)
x_all_scaled = scalerX.fit_transform(D_x.reshape(-1, 1))
x_all_poly = poly.fit_transform(x_all_scaled)

pipeline = make_pipeline(Ridge(alpha=0))
pipeline.fit(x_all_poly, scalerY.fit_transform(D_y.reshape(-1, 1)))

# Prepare test data for final predictions
x_test_final_scaled = scalerX.transform(testData[0].reshape(-1, 1))
x_test_final_poly = poly.transform(x_test_final_scaled)
test_predictions = pipeline.predict(x_test_final_poly)
FinalRMSETestingData = np.sqrt(mean_squared_error(testData[1], scalerY.inverse_transform(test_predictions)))

# Prepare training data for final predictions
train_predictions = pipeline.predict(x_all_poly)
FinalRMSETrainingData = np.sqrt(mean_squared_error(D_y, scalerY.inverse_transform(train_predictions)))

# Final Model Results
print("\n--- Final Model Results ---\n")

# Print final coefficients
coefficients = pipeline.named_steps['ridge'].coef_.flatten()
print(f"Final Coefficients for Polynomial Degree {best_poly_degree}:")
for idx, coef in enumerate(coefficients):
    print(f"Coefficient {idx}: {coef:.6f}")

# Print final RMSE values
print('\n')
print("Final RMSE Training Data:", FinalRMSETrainingData)
print("Final RMSE Testing Data:", FinalRMSETestingData)
print('\n')

# Plot the RMSE results
def plot_results(avgTrainErr, avgTestErr):
    degrees = range(13)
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, avgTrainErr, label='Train RMSE')
    plt.plot(degrees, avgTestErr, label='Test RMSE')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Root Mean Squared Error")
    plt.title("Train vs Test RMSE")
    plt.legend()
    plt.savefig('Train_vs_Test_RMSE.png')
    plt.show()

plot_results(avgTrainErr, avgTestErr)

# Plot the training data, predictions, and testing data
def plot_data_and_predictions(D_x, D_y, testData, train_predictions, test_predictions, degree):
    # Sort training data and predictions
    sorted_indices_train = np.argsort(D_x)
    sorted_x_train = D_x[sorted_indices_train]
    sorted_y_train = D_y[sorted_indices_train]
    sorted_train_predictions = train_predictions[sorted_indices_train]
    
    # Sort testing data and predictions
    sorted_indices_test = np.argsort(testData[0])
    sorted_x_test = testData[0][sorted_indices_test]
    sorted_y_test = testData[1][sorted_indices_test]
    sorted_test_predictions = test_predictions[sorted_indices_test]
    
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.scatter(D_x, D_y, color='blue', label='Training Data', s=50, edgecolor='k')
    
    # Plot testing data
    plt.scatter(testData[0], testData[1], color='green', label='Testing Data', s=50, edgecolor='k')
    
    # Plot predictions
    plt.plot(sorted_x_train, sorted_train_predictions, color='red', label=f'Polynomial Degree {degree} Predictions', linewidth=2)
    
    plt.xlabel("Year")
    plt.ylabel("Average Working Age")
    plt.title(f"Ridge Regression Predictions vs Data (Degree {degree})")
    plt.legend()
    plt.savefig(f'Ridge_Regression_Degree_{degree}.png')
    plt.show()

# Plot with the best polynomial degree
plot_data_and_predictions(D_x, D_y, testData, scalerY.inverse_transform(pipeline.predict(poly.transform(scalerX.transform(D_x.reshape(-1, 1))))), scalerY.inverse_transform(test_predictions), best_poly_degree)
