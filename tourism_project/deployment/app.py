import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Prediction App")
st.write("""
This application predicts the likelihood of a customer accepting tourism package based on their details.
Enter the data below to get a prediction.
""")

age    = st.number_input("Age", 18, 75, 30, 1)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", [1,2,3])
DurationOfPitch = st.number_input("Duration Of Pitch", 1, 150, 15, 1)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried","Large Business","Small Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", 1, 10, 2, 1)
NumberOfFollowups = st.number_input("Number Of Follow ups", 1, 10, 2, 1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe","Standard","Super Deluxe","King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3, 1)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single","Divorced","Unmarried"])
NumberOfTrips = st.number_input("Number Of Trips", 1, 20, 2, 1)
Passport = st.selectbox("Passport", [0,1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3, 1)
OwnCar = st.selectbox("Own Car", [0,1])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", 0, 10, 0, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager","Senior Manager","VP","AVP"])
MonthlyIncome = st.number_input("Monthly Income", 100, 100000, 5000, 100)



input_data = pd.DataFrame([{
    "Age" : age,
    "Type of Contact" : TypeofContact,
    "City Tier" : CityTier,
    "Duration Of Pitch" : DurationOfPitch,
    "Occupation" : Occupation,
    "Gender" : Gender,
    "Number Of Person Visiting" : NumberOfPersonVisiting,
    "Number Of Follow ups" : NumberOfFollowups,
    "Product Pitched" : ProductPitched,
    "Preferred Property Star" : PreferredPropertyStar,
    "Marital Status" : MaritalStatus,
    "Number Of Trips" : NumberOfTrips,
    "Passport" : Passport,
    "Pitch Satisfaction Score" : PitchSatisfactionScore,
    "Own Car" : OwnCar,
    "Number Of Children Visiting" : NumberOfChildrenVisiting,
    "Designation" : Designation,
    "Monthly Income" : MonthlyIncome,      
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction == 1 else "No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
