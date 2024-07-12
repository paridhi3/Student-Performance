# Import necessary libraries and modules
import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException  # Custom exception for handling errors

def save_object(file_path, obj):
    """
    Save an object (our model) to a file using pickle.
    
    file_path: Path where the object will be saved
    obj: Object to be saved
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception in case of any error
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models using GridSearchCV and return their test scores.
    
    Returns: Dictionary containing the test scores of each model
    """
    try:
        report = {}

        # Iterate over each model and its parameters
        for i in range(len(list(models))):

            model = list(models.values())[i]  # Get the model
            para = param[list(models.keys())[i]]  # Get the parameters

            # Perform GridSearchCV to find the best parameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the best parameters to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores for training and testing data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        # Raise a custom exception in case of any error
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using pickle.
    
    file_path: Path to the file containing the object
    Returns: Loaded object
    """
    try:
        # Load the object from the file
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        # Raise a custom exception in case of any error
        raise CustomException(e, sys)
