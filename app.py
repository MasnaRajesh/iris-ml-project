from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and target names
model_data = joblib.load('model.pkl')
model = model_data['model']
target_names = model_data['target_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Make a prediction
        prediction = model.predict(final_features)
        species = target_names[prediction[0]].capitalize()
        
        return render_template('index.html', prediction_text=f'Predicted Species: {species}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: Please enter valid numbers. ({e})')

if __name__ == "__main__":
    # Run the app, listening on all public IPs on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)