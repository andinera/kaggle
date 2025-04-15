import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")




# Tutorial - Random Forest

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
print(output)
output.to_csv('submission.csv', index=False)




# Linear Regression

from sklearn.linear_model import LinearRegression

y_train = train_data["Survived"]

features = ["Pclass", "Sex"]
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred_rounded}).astype(int)
output.to_csv('submission.csv', index=False)




# Linear Regression
# Split up train data into train/test to evaluate different models before submitting

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

y_train = train_data_80["Survived"]
y_test = test_data_20["Survived"]

features = ["Pclass", "Sex"]
X_train = pd.get_dummies(train_data_80[features])
X_test = pd.get_dummies(test_data_20[features])

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R²):", r2)




# Logistic Regression

