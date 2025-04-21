# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session






train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()








test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()








women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)






men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)








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
print("Your submission was successfully saved!")









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
print("Your submission was successfully saved!")










from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd

# # Sample data (replace with your actual data)
# data = {'feature1': [1, 2, 3, 4, 5], 
#         'feature2': [5, 4, 3, 2, 1],
#         'target':   [0, 1, 0, 1, 0]}
# df = pd.DataFrame(data)

# # Split data into features (X) and target (y)
# X = df[['feature1', 'feature2']]
# y = df['target']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a decision tree classifier
# dt_classifier = DecisionTreeClassifier()

# # Train the classifier
# dt_classifier.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = dt_classifier.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)