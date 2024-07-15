from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model/model_pipeline.pkl')
print(model)

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the form submission and predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        form_data = request.form.to_dict()

        # Debug: Print the form data
        print("Form Data:", form_data)
        
        # Prepare the data for prediction
        new_data = pd.DataFrame({
            'age': [int(form_data['age'])],
            'sex': [int(form_data['sex'])],
            'cp': [int(form_data['cp'])],
            'trtbps': [int(form_data['trtbps'])],
            'chol': [int(form_data['chol'])],
            'fbs': [int(form_data['fbs'])],
            'restecg': [int(form_data['restecg'])],
            'thalachh': [int(form_data['thalachh'])],
            'exng': [int(form_data['exng'])],
            'oldpeak': [float(form_data['oldpeak'])],
            'slp': [int(form_data['slp'])],
            'caa': [int(form_data['caa'])],
            'thall': [int(form_data['thall'])]
        })

        # Debug: Print the DataFrame
        print("DataFrame Columns:", new_data.columns)

        # Make prediction
        prediction = model.predict(new_data)
        
        # Decode the prediction
        prediction_label = 'Less chance of heart attack' if prediction[0] == 1 else 'More chance of heart attack'

# Return prediction as JSON
        return jsonify({'result': prediction_label})
    
    except KeyError as e:
        return jsonify({'error': f"Missing form data for: {e.args[0]}"})

if __name__ == '__main__':
    app.run(debug=True)
