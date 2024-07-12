# Import necessary libraries and modules
import os
import sys
import pandas as pd
from src.exception import CustomException  # Custom exception for handling errors
from src.utils import load_object  # Utility function to load saved objects


class CustomData:
    """
    A class to handle custom input data entered by the user.
    """
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        
        # Initialize the custom data attributes.
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Convert the custom data attributes (entered by user) into a DataFrame.
        Returns: DataFrame containing the custom data
        """
        try:
            # Create a dictionary from the custom data attributes
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary to a DataFrame and return it
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise a custom exception in case of any error
            raise CustomException(e, sys)


class PredictPipeline:
    """
    A class to handle the prediction pipeline.
    Contains methods to generate predictions based on user input values.
    """

    def __init__(self):
        pass

    def predict(self, features):
        """
        Load the model and preprocessor, then predict the target using the provided features by the user.

        features: DataFrame containing the input features provided by the user
        Returns: Array of predictions
        """
        try:
            # Define paths to the saved model and preprocessor objects
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading")

            # Load the saved model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict the target using the loaded model
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            # Raise a custom exception in case of any error
            raise CustomException(e, sys)
