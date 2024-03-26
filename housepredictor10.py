import streamlit as st
import pickle

## Deserialise - unpacking pickle
with open('C:/Users/Hasan/OneDrive/Documents/Digitalfutures/Machine Learning/rf_houses.sav', 'rb') as file: 
    rf_loaded = pickle.load(file) 
file.close()

import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Define the options for each feature
property_type_options = {'Flats/Maisonettes': 'Flats/Maisonettes', 'Other': 'Other', 'Semi-Detached': 'Semi-Detached', 'Terraced': 'Terraced', 'Detached': 'Detached'}
old_new_options = {'New': 'New', 'Old': 'Old'}
duration_options = {'Freehold': 'Freehold', 'Leasehold': 'Leasehold'}
district_options = ['BARNET', 'BEXLEY', 'BRENT', 'BROMLEY', 'CAMDEN', 'CITY OF LONDON', 'CITY OF WESTMINSTER',
                    'CROYDON', 'EALING', 'ENFIELD', 'GREENWICH', 'HACKNEY', 'HAMMERSMITH AND FULHAM', 'HARINGEY',
                    'HARROW', 'HAVERING', 'HILLINGDON', 'HOUNSLOW', 'ISLINGTON', 'KENSINGTON AND CHELSEA',
                    'KINGSTON UPON THAMES', 'LAMBETH', 'LEWISHAM', 'MERTON', 'NEWHAM', 'REDBRIDGE',
                    'RICHMOND UPON THAMES', 'SOUTHWARK', 'SUTTON', 'TOWER HAMLETS', 'WALTHAM FOREST', 'WANDSWORTH']

# Create the Streamlit app
def main():
    st.title('Property Information')

    # Collect user inputs
    property_type = st.selectbox('Select Property Type', options=list(property_type_options.keys()))
    old_new = st.radio('Old/New', options=list(old_new_options.keys()))
    duration = st.radio('Duration', options=list(duration_options.keys()))
    district = st.selectbox('Select District', options=district_options)

    # One-hot encode the user inputs
    property_type_encoded = encode_property_type(property_type)
    old_new_encoded = encode_old_new(old_new)
    duration_encoded = encode_duration(duration)
    district_encoded = encode_district(district)

    # Create DataFrame with encoded values
    data = {
        'Old/New': [old_new_encoded],
        'Duration': [duration_encoded],
        'ptype_F': [property_type_encoded['Flats/Maisonettes']],
        'ptype_O': [property_type_encoded['Other']],
        'ptype_S': [property_type_encoded['Semi-Detached']],
        'ptype_T': [property_type_encoded['Terraced']]
    }
    data.update({f'dis_{d}': [district_encoded[d]] for d in district_options})
    df = pd.DataFrame(data)

    # Now, you can use this DataFrame to make predictions with the loaded model
    predictions = rf_loaded.predict(df)

    # Do something with the predictions
    st.write(predictions)

    # Display the DataFrame
    st.write('DataFrame:', df)

def encode_property_type(property_type):
    encoder = OneHotEncoder(categories=[list(property_type_options.keys())], sparse=False)
    encoded = encoder.fit_transform([[property_type]])
    return {key: encoded[0][i] for i, key in enumerate(property_type_options.keys())}

def encode_old_new(old_new):
    return 1 if old_new == 'New' else 0

def encode_duration(duration):
    return 1 if duration == 'Freehold' else 0

def encode_district(district):
    return {d: 1 if d == district else 0 for d in district_options}

if __name__ == "__main__":
    main()


    
