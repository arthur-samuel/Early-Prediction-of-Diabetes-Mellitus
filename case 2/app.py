import gradio as gr
import pandas as pd
import pickle
import sklearn

# Load the saved model from the pickle file
with open('logistic_regression_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Define the input interface
def diabetes_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    data = {
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
        "Age": [Age]
    }
    df = pd.DataFrame.from_dict(data)
    outcome = model.predict(df)
    return outcome[0]

# Define multiple examples for the input interface
examples = [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    [8, 183, 64, 0, 0, 23.3, 0.672, 32],
    [1, 89, 66, 23, 94, 28.1, 0.167, 21],
    [0, 137, 40, 35, 168, 43.1, 2.288, 33]
]

# Create the input interface and launch the app
gr.Interface(fn=diabetes_prediction, 
             inputs=["number", "number", "number", "number", "number", "number", "number", "number"], 
             outputs="number", 
             title="Diabetes Prediction Model", 
             description="Enter the following features to predict the likelihood of diabetes.",
             examples=examples,
             inline=False, 
             debug=True).launch()
