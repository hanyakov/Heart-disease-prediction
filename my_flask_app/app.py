from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/model_pipeline.pkl')

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and predict
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    form_data = request.form.to_dict()
    
    # Prepare the data for prediction
    new_data = pd.DataFrame({
        'age': [form_data['age']],
        'sex': [form_data['sex']],
        'cp': [form_data['cp']],
        'trtbps': [form_data['trtbps']],
        'chol': [form_data['chol']],
        'fbs': [form_data['fbs']],
        'restecg': [form_data['restecg']],
        'thalach': [form_data['thalach']],
        'exng': [form_data['exng']],
        'oldpeak': [form_data['oldpeak']],
        'slope': [form_data['slope']],
        'ca': [form_data['ca']],
        'thal': [form_data['thal']],
        'output': [form_data['output']]
    })

    # Make prediction
    prediction = model.predict(new_data)
    
    # Decode the prediction
    prediction_label = 'Less chance of heart attack ' if prediction[0] == 1 else 'More chance of heart attack'

    # Render the result page with prediction
    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
