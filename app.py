import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.title("Smart Lender - Loan Approval Prediction")

# Load dataset
df = pd.read_csv("loan_data.csv")
df.fillna(df.mode().iloc[0], inplace=True)
df = pd.get_dummies(df)

X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# User Input
st.sidebar.header("Applicant Details")
income = st.sidebar.number_input("Applicant Income")
loan_amt = st.sidebar.number_input("Loan Amount")
credit = st.sidebar.selectbox("Credit History", [0,1])

if st.button("Predict Loan Status"):
    prediction = model.predict([[income, loan_amt, credit]])
    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
