# Import necessary libraries and modules
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)

# Alias for the Flask application
app = application

# Route for the home page
@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        # Create an instance of CustomData with the form input values entered by user
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        # Convert the input data into a DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Create an instance of the prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        # Perform prediction
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        # Render the results on the form page
        return render_template('home.html', results=results[0])

# Main entry point for running the Flask application
if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0")
