import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Projects/trained_model.sav', 'rb'))


# Diabetes prediction predictive system
input_data = (5,116,74,0,0,25.6,0.201,30)

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction[0])

if(prediction[0] == 1):
    print("\nThe person is Diabetic\n")
else:
    print("\nThe person is Non-Diabetic\n")