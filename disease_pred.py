import numpy as np
import pandas as pd
import sklearn
df = pd.read_csv("heart_disease_.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Patients without heart problems: ", countNoDisease)
print("Patients with heart problems: ", countHaveDisease)

X = df.iloc[:,:-1] # Independent Variables
y = df.iloc[:,-1] # Dependent Variables
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
# Standardize the feature set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print (sklearn.metrics.accuracy_score (y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    
}

scoring = ['accuracy', 'precision', 'recall', 'f1'] 
results = []

import time
from time import time
# Train and evaluate each model using K-fold cross-validation
from sklearn.model_selection import cross_validate
for name, model in models.items(): 
    start_time = time() # Start the timer
    scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
    end_time=time()

    accuracy = np.mean(scores['test_accuracy'])
    precision = np.mean(scores['test_precision'])
    recall = np.mean(scores['test_recall'])
    f1 = np.mean(scores['test_f1'])
    elapsed_time = end_time - start_time # Time taken to run the model

    results.append({
        'Model': name,
        'Model()':model,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Time (s)': elapsed_time
    })

    print ("completed in ", elapsed_time ,"seconds.")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Accuracy", ascending=False)
print(results_df)
b_model = results_df.iloc[0]
f_model = b_model['Model()']
acc_model = b_model ["Accuracy"]
recal = b_model ["Recall"]
F_one = b_model ["F1-score"]
print ("Best Model : ", f_model)
print ("Model Accuracy : " , acc_model)
print ("Model Recall : " , recal)
print ("Model F1 score : " , F_one)
f_model.fit(X_train , y_train)

print ("-------------------------------------------------------------------------------")
print ("--------------------------- User Prediction Section ---------------------------")

age=int(input("Enter age : "))
sex=int(input("Enter sex (Enter 1 for male | 0 for female): "))
cp=int(input("Enter chest pain type : "))
trestbps=int(input("Enter Resting Blood Pressure : "))
chol=int(input("Enter Cholestrol : "))
fbs=int(input("Enter fasting Blood Sugar : "))
restecg=int(input("Enter Resting ECG : "))
thalach=int(input("Enter Max Heart Rate : "))
exang=int(input("Enter exercise induced angina : "))
oldpeak=float(input("Enter ST depression (exercise) : "))
slope=int(input("Enter slope of peak exercise (ST) : "))
ca=int(input("Enter number of major vessels : "))
thal=int(input("Enter Thalassemia : "))

input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
input_data_as_df = pd.DataFrame([input_data], columns=X.columns)
input_data_scaled = scaler.transform(input_data_as_df)
prediction = f_model.predict(input_data_scaled)
print (prediction)
if (prediction[0]==0):
    print('the person doesnt have disease')
else:
    print('the person does have disease')

    