import pickle
import numpy as np

# load the model from disk
filename = 'finalized_model.pkl'

loaded_model = pickle.load(open(filename, 'rb'))

def predict_diabetes(input):
    for i in range(1, 7):
        if input[i] == 0:
            input[i] = np.nan
    print("The prediction is {}.".format(loaded_model.predict([input])[0]))

input = [
        7,  # Pregnancies
        161,  # Glucose
        86,  # BloodPressure
        0,  # SkinThickness
        0,  # Insulin
        30.4,  # BMI
        0.165,  # DiabetesPedigreeFunction
        47,  # Age
    ]
predict_diabetes(input)