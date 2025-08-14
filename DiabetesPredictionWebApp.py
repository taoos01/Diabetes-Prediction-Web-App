import numpy as np
import pickle # Used for loading the saved model
import streamlit as st # Used for model deployment
# rb -> Read Binary

# Loading the saved model
# pickle.load(directory ofthe file)
# '\' should be replaced by '/' because sometime it may not run properly
loaded_model = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Projects/trained_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):
    # input_data = (5,116,74,0,0,25.6,0.201,30)

    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction[0])

    if(prediction[0] == 1):
        return ("The person is Diabetic")
    else:
        return ("The person is Non-Diabetic")
    
def main():
    # Title for our webpage
    st.title("Diabetes Prediction Web App")

    # Getting the input data from the user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    # Taking input of all the feature
    Pregnancies = st.text_input("No of pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("Body Mass Index value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age of the person")

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    # st.button("Button Name")
    if(st.button("Diabetes Test Result")):
        diagnosis = diabetes_prediction([float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()