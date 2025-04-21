import numpy as np
import pandas as pd

train_data = pd.read_csv("./train.csv")


from sklearn.model_selection import train_test_split

features = ["Pclass", "Sex", "Age", "SibSp"]
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
X = pd.get_dummies(train_data[features])
y = train_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

model = AdaBoostClassifier(n_estimators=1000, algorithm='SAMME.R')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred)

accuracy = accuracy_score(y_test, y_pred_rounded)
print("Accuracy:", accuracy)