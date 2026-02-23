from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

X = np.array([[80,100],[120,250],[70,90],[130,300]])
y = np.array([0,1,0,1])

model = LogisticRegression()
model.fit(X,y)

joblib.dump(model,"health_model.pkl")

print("Model Created Successfully")