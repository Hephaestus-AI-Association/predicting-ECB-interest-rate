import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Step 1: Load the Data
data = pd.read_csv("output_file.csv")

# Step 2: Prepare Data
X = data[['Inflation','BCI','LT','ST']] # Features
y = data['Interest_Rate'] # Target variable

# Step 3: Split Data
kf = KFold(n_splits=5, shuffle=True, random_state=60)

scores = []
mse_scores = []
mae_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create an SVR model
    svr = SVR(kernel='rbf', gamma=0.1, epsilon=0.05)

    # Fit the model to the training data
    svr.fit(X_train_scaled, y_train)

    # Predict on the testing data
    y_pred = svr.predict(X_test_scaled)

    # Calculate the R-squared, mse and mae scores
    r_squared = r2_score(y_test, y_pred)
    scores.append(r_squared)

    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

    # Print dates along with actual and predicted values (not very relevant)
    print(f"\nFold {len(scores)} - Dates:")
    print(data['Date'].iloc[test_index].values)
    print("\nActual Interest Rates:")
    print(y_test.values)
    print("\nPredicted Interest Rates:")
    print(y_pred)

    # Plot actual vs predicted values for each fold with lines and markers
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'].iloc[test_index], y_test, marker='o', linestyle='-', label='Actual Interest Rate', color='blue')
    plt.plot(data['Date'].iloc[test_index], y_pred, marker='o', linestyle='-', label='Predicted Interest Rate', color='red')
    plt.xlabel('Date')
    plt.ylabel('Interest Rate')
    plt.title(f'Fold {len(scores)}: Actual vs Predicted Interest Rates')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Boxplot of R-squared scores
plt.boxplot(scores)
plt.xlabel("Fold Number")
plt.ylabel("R-squared Score")
plt.title("Boxplot of R-squared Scores")
plt.show()

# Bar plot of R-squared scores for each fold
plt.bar(range(1, len(scores)+1), scores)
plt.xlabel("Fold Number")
plt.ylabel("R-squared Score")
plt.title("R-squared Scores for Each Fold")
plt.show()

print("R-squared scores:", scores)
print("Mean Squared Error scores:", mse_scores)
print("Mean Absolute Error scores:", mae_scores)

print("Mean R-squared:", np.mean(scores))
print("Standard Deviation of R-squared:", np.std(scores))

# Scatter plot of predicted vs. actual interest rates (better present it)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Interest Rate")
plt.ylabel("Predicted Interest Rate")
plt.title("Scatter Plot of Predicted vs. Actual Interest Rates")
plt.plot([0, 4.5], [0, 4.5], color='r')
plt.show()

# Residual plot (to explain in overleaf)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Interest Rate")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Learning Curve (to explain in overleaf)
train_sizes, train_scores, test_scores = learning_curve(
    svr, X_train_scaled, y_train, cv=kf, scoring='r2', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training Set Size')
plt.ylabel('R-squared Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.show()