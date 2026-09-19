import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.joblib')

def predict_salary(years_experience):

    prediction = model.predict([[years_experience]])
    return float(prediction[0])


demo = gr.Interface(
    fn=predict_salary,
    inputs=gr.Number(label="Years of Experience", minimum=0, value=1.0),
    outputs=gr.Number(label="Predicted Salary"),
    title="Salary Predictor",
    description="Enter your years of experience to predict your expected salary using our trained Linear Regression model."
)

if __name__ == "__main__":
    demo.launch()
