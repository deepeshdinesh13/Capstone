# Import the required libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Setting a theme for seaborn plots
sns.set_theme(style="whitegrid")

# Streamlit app title
st.title('Real Estate Listings Analysis')
df = pd.read_csv('HouseListings-Top45Cities-10292023-kaggle.csv', encoding='latin1')
# Showing a random sample of the data with a unique key for the checkbox
if st.checkbox('Show random sample of data before pre-processing', key='show_old_sample'):
    st.write(df.sample(5))

# Load the data with caching to speed up app loading
@st.cache
def load_data():
    df = pd.read_csv('HouseListings-Top45Cities-10292023-kaggle.csv', encoding='latin1')
    df.drop_duplicates(inplace=True)
    df.drop(['Address', 'Latitude', 'Longitude'], axis=1, inplace=True)
    # Dropping non-sensible rows
    df = df[df['Number_Beds'] <= 20]
    df = df[df['Number_Baths'] <= 10]
    df = df[~((df['Number_Baths'] == 0) & (df['Number_Beds'] == 0))]
    # Replacing values for bedrooms and bathrooms
    df.loc[df['Number_Beds'] > 4, 'Number_Beds'] = '>4'
    df.loc[df['Number_Baths'] > 3, 'Number_Baths'] = '>3'
    # Shortening province names
    long_to_short = {
        'Ontario': 'ON', 'British Columbia': 'BC', 'Alberta': 'AB', 
        'Saskatchewan': 'SK', 'Newfoundland and Labrador': 'NL', 
        'New Brunswick': 'NB', 'Quebec': 'QC', 'Manitoba': 'MB', 'Nova Scotia': 'NS'
    }
    df.replace(long_to_short, inplace=True)
    return df

df = load_data()

# Showing a random sample of the data with a unique key for the checkbox
if st.checkbox('Show random sample of data after pre-processing', key='show_sample'):
    st.write(df.sample(5))

# Toggle for Histogram & KDE of House Prices
if st.checkbox('Show Histogram & KDE of House Prices'):
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], kde=True, bins=30, ax=ax)
    ax.set_title('Histogram & KDE of Price')
    st.pyplot(fig)

# Toggle for Histogram & KDE of Median Family Income
if st.checkbox('Show Histogram & KDE of Median Family Income'):
    fig, ax = plt.subplots()
    sns.histplot(df['Median_Family_Income'], kde=True, bins=20, ax=ax)
    ax.set_title('Histogram & KDE of Median Family Income')
    st.pyplot(fig)

# Toggle for Median Family Income and House Price per City (Top 15)
if st.checkbox('Show Median Family Income and House Price per City (Top 15)'):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    df.groupby('City')['Median_Family_Income'].mean().head(15).sort_values().plot(kind='barh', ax=ax[0])
    df.groupby('City')['Price'].mean().head(15).sort_values().plot(kind='barh', ax=ax[1])
    st.pyplot(fig)

# Toggle for Median Family Income and House Price per Province
if st.checkbox('Show Median Family Income and House Price per Province'):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    df.groupby('Province')['Median_Family_Income'].mean().sort_values().plot(kind='barh', ax=ax[0])
    df.groupby('Province')['Price'].mean().sort_values().plot(kind='barh', ax=ax[1])
    st.pyplot(fig)

# Toggle for House Prices by Number of Beds per Province
if st.checkbox('Show House Prices by Number of Beds per Province'):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(data=df, x='Province', y='Price', hue='Number_Beds', palette="Blues", ax=ax)
    ax.set_title('Price of House Listings According to the Number of Beds per Province')
    st.pyplot(fig)

# Toggle for House Prices by Number of Bathrooms per Province
if st.checkbox('Show House Prices by Number of Bathrooms per Province'):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.barplot(data=df, x='Province', y='Price', hue='Number_Baths', hue_order=['0', '1', '2', '3', '>3'], palette="Blues", ax=ax)
    ax.set_title('Price of House Listings According to the Number of Bathrooms per Province')
    st.pyplot(fig)
    
@st.cache(suppress_st_warning=True)



def process_and_train_models():
    # Load and preprocess data
    df = pd.read_csv('HouseListings-Top45Cities-10292023-kaggle.csv', encoding='latin1')
    df.drop_duplicates(inplace=True)
    df.drop(['Address', 'Latitude', 'Longitude'], axis=1, inplace=True)
    df = df[df['Number_Beds'] <= 20]
    df = df[df['Number_Baths'] <= 10]
    df = df[~((df['Number_Baths'] == 0) & (df['Number_Beds'] == 0))]
    df['Number_Beds'] = np.where(df['Number_Beds'] > 4, '>4', df['Number_Beds'].astype(str))
    df['Number_Baths'] = np.where(df['Number_Baths'] > 3, '>3', df['Number_Baths'].astype(str))
    
    # Splitting the data
    y = df['Price']
    X = df.drop('Price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define ColumnTransformer for preprocessing
    ct = ColumnTransformer(transformers=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), ['City', 'Province']),
        ('oe', OrdinalEncoder(), ['Number_Beds', 'Number_Baths']),
        ('scaler', StandardScaler(), ['Population', 'Median_Family_Income'])
    ], remainder='passthrough')
    
    # Apply transformations
    X_train_transformed = ct.fit_transform(X_train)
    X_test_transformed = ct.transform(X_test)
    
    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_transformed, y_train)
    y_pred = lr.predict(X_test_transformed)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Visualization: Actual vs. Predicted Values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs. Predicted Values')
    
    # Display the evaluation metrics and plot in Streamlit
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    st.pyplot(fig)
    
    # Comparison DataFrame
    comparison_df = pd.DataFrame({'Actual Value': y_test.iloc[:10].values, 'Predicted Value': y_pred[:10]})
    st.write("Comparison of Actual vs. Predicted Values:", comparison_df)

# Calling the function within Streamlit
if st.button('Run Model'):
    process_and_train_models()