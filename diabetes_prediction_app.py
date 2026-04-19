import numpy as np
import pickle
import streamlit as st
import os

# loading the saved model (relative path — same folder as this script)
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))


# creating a function for Prediction
def diabetes_prediction(input_data):

    # changing the input_data to numpy array (as floats)
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is **not diabetic** ✅'
    else:
        return 'The person is **diabetic** ⚠️'


def main():

    # Page config
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

    # giving a title
    st.title('🩺 Diabetes Prediction Web App')
    st.markdown('Enter the patient details below and click **Predict** to check for diabetes.')
    st.markdown('---')

    # getting the input data from the user in two columns
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
        Glucose = st.number_input('Glucose Level', min_value=0.0, max_value=300.0, value=100.0, step=1.0)
        BloodPressure = st.number_input('Blood Pressure value', min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        SkinThickness = st.number_input('Skin Thickness value', min_value=0.0, max_value=100.0, value=20.0, step=1.0)

    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        BMI = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        Age = st.number_input('Age of the Person', min_value=1, max_value=120, value=30, step=1)

    st.markdown('---')

    # creating a button for Prediction
    if st.button('🔍 Diabetes Test Result', use_container_width=True):
        diagnosis = diabetes_prediction([
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ])
        st.subheader('Result:')
        if 'not diabetic' in diagnosis:
            st.success(diagnosis)
        else:
            st.warning(diagnosis)


if __name__ == '__main__':
    main()