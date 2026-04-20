import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model():
    # Load dataset
    df = pd.read_csv('sales_employee_data.csv')
    
    # Simple feature selection
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'sales_model.sav')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {model_path}")

if __name__ == '__main__':
    train_model()
