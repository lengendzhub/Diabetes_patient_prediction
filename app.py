import numpy as np
import pickle
import streamlit as st
import os
import pandas as pd

# Page config
st.set_page_config(page_title="Sales Employee Analysis", page_icon="📈", layout="wide")

# loading the saved model
model_path = os.path.join(os.path.dirname(__file__), 'sales_model.sav')

def load_model():
    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))
    return None

loaded_model = load_model()

# prediction function
def predict_attrition(input_data):
    if loaded_model is None:
        return "Model not found. Please train the model first."
    
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'Likely to **Stay** ✅'
    else:
        return 'High Risk of **Attrition** ⚠️'

def main():
    st.title('📈 Sales Employee Analysis & Prediction')
    st.markdown("""
    This application analyzes sales employee data and predicts the risk of attrition based on performance and engagement metrics.
    """)
    st.divider()

    # Sidebar for data overview
    st.sidebar.header("Data Overview")
    if os.path.exists('sales_employee_data.csv'):
        df = pd.read_csv('sales_employee_data.csv')
        st.sidebar.write(f"Total Employees: {len(df)}")
        st.sidebar.write(f"Average Monthly Sales: ${df['Monthly_Sales_USD'].mean():,.2f}")
        if st.sidebar.checkbox("Show raw data"):
            st.write(df.head())
    else:
        st.sidebar.warning("Dataset not found.")

    st.subheader("🔍 Predict Employee Attrition Risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', min_value=18, max_value=70, value=30)
        years_at_company = st.number_input('Years at Company', min_value=0, max_value=40, value=5)
        monthly_income = st.number_input('Monthly Income ($)', min_value=1000, max_value=50000, value=5000)

    with col2:
        monthly_sales = st.number_input('Monthly Sales ($)', min_value=0, max_value=200000, value=15000)
        total_working_years = st.number_input('Total Working Years', min_value=0, max_value=50, value=8)
        training_times = st.number_input('Training Times (Last Year)', min_value=0, max_value=10, value=2)

    with col3:
        work_life_balance = st.slider('Work-Life Balance (1-4)', 1, 4, 3)
        performance_rating = st.slider('Performance Rating (1-4)', 1, 4, 3)

    st.divider()
    
    if st.button('Predict Attrition Risk', use_container_width=True):
        result = predict_attrition([
            age, years_at_company, monthly_income, monthly_sales,
            total_working_years, training_times, work_life_balance, performance_rating
        ])
        
        st.subheader("Prediction Result:")
        if 'Stay' in result:
            st.success(result)
        else:
            st.error(result)

if __name__ == '__main__':
    main()
