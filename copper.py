import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Load the models and scalers
with open('rf_regressor_model.pkl', 'rb') as f:
    rf_regressor = pickle.load(f)

#with open('gb_regressor_model.pkl', 'rb') as f:
    #gb_regressor = pickle.load(f)

with open('et_classifier_model.pkl', 'rb') as f:
    et_classifier = pickle.load(f)

#with open('xgb_classifier_model.pkl', 'rb') as f:
    #xgb_classifier = pickle.load(f)

#with open('log_classifier_model.pkl', 'rb') as f:
    #logistic_regression = pickle.load(f)


# Streamlit application
st.title('Predictive Modeling Application')

model_type = st.selectbox("Select model type:", ["Regression", "Classification"])

if model_type == "Regression":
    st.header("Predict Selling Price")

    # Input fields for regression
    customer_id= st.text_input("Customer ID")
    customer=st.text_input("Customer")
    country = st.text_input("Country")
    item_type = st.text_input("Item Type")
    application = st.text_input("Application")
    thickness = st.number_input("Thickness", format="%.2f")
    width = st.number_input("Width", format="%.2f")
    item_date = st.date_input("Item Date")
    quantity_tons = st.number_input("Quantity (tons)", format="%.2f")
    status = st.selectbox("Status", ["Draft", "Won", "Lost"])
    material_ref = st.text_input("Material Reference")
    product_ref = st.number_input("Product Reference", format="%.0f")
    delivery_date = st.date_input("Delivery Date")

    # Convert inputs to appropriate types and preprocess
    input_data = {
        'id': ["customer_id"],  # Placeholder for ID generation logic
        'item_date': [item_date.strftime('%Y%m%d')],
        'quantity tons': [float(quantity_tons)],
        'customer': [customer],
        'country': [country],
        'status': [status],
        'item type': [item_type],
        'application': [(application)],
        'thickness': [float(thickness)],
        'width': [float(width)],
        'material_ref': [material_ref],
        'product_ref': [int(product_ref)],
        'delivery date': [delivery_date.strftime('%Y%m%d')]
    }

    # Convert to DataFrame for further processing
    # Convert to DataFrame
    input_data_df = pd.DataFrame(input_data)

    categorical_columns = input_data_df.select_dtypes(include=['object']).columns

    reference_columns = ['material_ref', 'product_ref']
    for col in reference_columns:
        input_data_df[col] = input_data_df[col].astype('category')

    if 'index' in input_data_df.columns:
        input_data_df = input_data_df.drop(columns=['index'])

    numerical_columns = input_data_df.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if input_data_df[col].isnull().sum() > 0:
            input_data_df[col].fillna(input_data_df[col].median(), inplace=True) 

    categorical_columns = input_data_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if input_data_df[col].isnull().sum() > 0:
            input_data_df[col].fillna(input_data_df[col].mode()[0], inplace=True)
    
    #Identify and Remove Outliers using Isolation Forest
    from sklearn.ensemble import IsolationForest

    def remove_outliers_isolation_forest(df, columns):
        iso_forest = IsolationForest(contamination=0.01)  # Adjust contamination as needed
        for col in columns:
            if df[col].dtype != 'object':
                df[col] = df[col].fillna(df[col].median())
                df['outlier'] = iso_forest.fit_predict(df[[col]])
                df = df[df['outlier'] == 1]
                df = df.drop(columns=['outlier'])
        return df

    # Numerical columns to check for outliers
    numerical_columns = input_data_df.select_dtypes(include=[np.number]).columns
    input_data_df = remove_outliers_isolation_forest(input_data_df, numerical_columns)

    # Convert dates to datetime
    input_data_df['item_date'] = pd.to_datetime(input_data_df['item_date'])
    input_data_df['delivery date'] = pd.to_datetime(input_data_df['delivery date'])

    # Create new feature: days_to_delivery
    input_data_df['days_to_delivery'] = (input_data_df['delivery date'] - input_data_df['item_date']).dt.days

    # Drop original date columns
    input_data_df.drop(['item_date', 'delivery date'], axis=1, inplace=True)

    # Convert '00000' values to NaN in 'material_ref' column
    input_data_df['material_ref'] = input_data_df['material_ref'].replace('00000', np.nan)



    if 'status'=='Won':
        input_data_df['status']=7
    elif 'status'=="Lost":
        input_data_df['status']=3
    else:
        input_data_df['status']=1

    if 'item type'=='W':
        input_data_df['item type']=3
    elif 'item type'=="Lost":
        input_data_df['item type']=5
    else:
        input_data_df['item type']=1
    


    # Drop columns that are not needed for correlation analysis
    input_data_df.drop(['id', 'quantity tons', 'material_ref', 'product_ref'], axis=1, inplace=True)

    # Check for and replace infinite values with NaN
    input_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)


    # Convert all columns to float
    #input_data_df = input_data_df.astype(float)

    # Define a threshold for extremely large values
    threshold = 1e+10

    # Clip values larger than the threshold
    input_data_df[input_data_df.select_dtypes(include=[np.number]) > threshold] = threshold

    
    if st.button("Predict Selling Price"):
        input_data_scaled = input_data_df  
        rf_prediction = rf_regressor.predict(input_data_scaled)[0]
        
        # Display the result in the middle of the screen with bold and larger font
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
            <h1 style="font-size: 3em; font-weight: bold;">Selling Price: ${rf_prediction:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

elif model_type == "Classification":
    st.header("Predict Status")

# Input fields for regression
    customer_id= st.text_input("Customer ID")
    customer=st.text_input("Customer")
    country = st.text_input("Country")
    item_type = st.text_input("Item Type")
    application = st.text_input("Application")
    thickness = st.number_input("Thickness", format="%.2f")
    width = st.number_input("Width", format="%.2f")
    item_date = st.date_input("Item Date")
    quantity_tons = st.number_input("Quantity (tons)", format="%.2f")
    material_ref = st.text_input("Material Reference")
    product_ref = st.number_input("Product Reference", format="%.0f")
    delivery_date = st.date_input("Delivery Date")
    selling_price = st.number_input("Selling Price", format="%.2f")

    # Convert inputs to appropriate types and preprocess
    input_data = {
        'id': ["customer_id"],  # Placeholder for ID generation logic
        'item_date': [item_date.strftime('%Y%m%d')],
        'quantity tons': [float(quantity_tons)],
        'customer': [customer],
        'country': [country],
        'item type': [item_type],
        'application': [(application)],
        'thickness': [float(thickness)],
        'width': [float(width)],
        'material_ref': [material_ref],
        'product_ref': [int(product_ref)],
        'delivery date': [delivery_date.strftime('%Y%m%d')],
        'selling_price': [float(selling_price)]
    }

    input_data_df = pd.DataFrame(input_data)

    categorical_columns = input_data_df.select_dtypes(include=['object']).columns

    reference_columns = ['material_ref', 'product_ref']
    for col in reference_columns:
        input_data_df[col] = input_data_df[col].astype('category')

    if 'index' in input_data_df.columns:
        input_data_df = input_data_df.drop(columns=['index'])

    numerical_columns = input_data_df.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if input_data_df[col].isnull().sum() > 0:
            input_data_df[col].fillna(input_data_df[col].median(), inplace=True) 

    categorical_columns = input_data_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if input_data_df[col].isnull().sum() > 0:
            input_data_df[col].fillna(input_data_df[col].mode()[0], inplace=True)
    
    #Identify and Remove Outliers using Isolation Forest
    from sklearn.ensemble import IsolationForest

    def remove_outliers_isolation_forest(df, columns):
        iso_forest = IsolationForest(contamination=0.01)  # Adjust contamination as needed
        for col in columns:
            if df[col].dtype != 'object':
                df[col] = df[col].fillna(df[col].median())
                df['outlier'] = iso_forest.fit_predict(df[[col]])
                df = df[df['outlier'] == 1]
                df = df.drop(columns=['outlier'])
        return df

    # Numerical columns to check for outliers
    numerical_columns = input_data_df.select_dtypes(include=[np.number]).columns
    input_data_df = remove_outliers_isolation_forest(input_data_df, numerical_columns)

    # Convert dates to datetime
    input_data_df['item_date'] = pd.to_datetime(input_data_df['item_date'])
    input_data_df['delivery date'] = pd.to_datetime(input_data_df['delivery date'])

    # Create new feature: days_to_delivery
    input_data_df['days_to_delivery'] = (input_data_df['delivery date'] - input_data_df['item_date']).dt.days

    # Drop original date columns
    input_data_df.drop(['item_date', 'delivery date'], axis=1, inplace=True)

    # Convert '00000' values to NaN in 'material_ref' column
    input_data_df['material_ref'] = input_data_df['material_ref'].replace('00000', np.nan)


    

    if 'item type'=='W':
        input_data_df['item type']=3
    elif 'item type'=="Lost":
        input_data_df['item type']=5
    else:
        input_data_df['item type']=1


    # Drop columns that are not needed for correlation analysis
    input_data_df.drop(['id', 'quantity tons', 'material_ref', 'product_ref'], axis=1, inplace=True)

    # Check for and replace infinite values with NaN
    input_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)


    # Convert all columns to float
    #input_data_df = input_data_df.astype(float)

    # Define a threshold for extremely large values
    threshold = 1e+10

    # Clip values larger than the threshold
    input_data_df[input_data_df.select_dtypes(include=[np.number]) > threshold] = threshold




    if st.button("Predict Status"):
        input_data_scaled = input_data_df  
        et_prediction = et_classifier.predict(input_data_scaled)[0]
        if et_prediction == 7:
            status_label = "WON"
        else:
            status_label = "LOST"
        

        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 200px;">
            <h1 style="font-size: 3em; font-weight: bold;">Status: {status_label}</h1>
        </div>
        """, unsafe_allow_html=True)

