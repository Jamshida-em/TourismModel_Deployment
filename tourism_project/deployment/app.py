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

Age_group    = st.selectbox("Age Group", ["18-25","26-41","42-57","58-76+"])
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", [1,2,3])
DurationOfPitch = st.number_input("Duration Of Pitch", 1, 150, 15, 1)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried","Large Business","Small Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", 1, 10, 2, 1)
NumberOfFollowups = st.number_input("Number Of Follow ups", 1, 10, 2, 1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe","Standard","Super Deluxe","King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", 1, 5, 3, 1)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single","Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", 1, 20, 2, 1)
Passport = st.selectbox("Passport", [0,1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3, 1)
OwnCar = st.selectbox("Own Car", [0,1])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", 0, 10, 0, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager","Senior Manager","VP","AVP"])
MonthlyIncome = st.number_input("Monthly Income", 100, 100000, 5000, 100)



input_data = pd.DataFrame([{
    "Age_group" : Age_group,
    "TypeofContact" : TypeofContact,
    "CityTier" : CityTier,
    "DurationOfPitch" : DurationOfPitch,
    "Occupation" : Occupation,
    "Gender" : Gender,
    "NumberOfPersonVisiting" : NumberOfPersonVisiting,
    "NumberOfFollowups" : NumberOfFollowups,
    "ProductPitched" : ProductPitched,
    "PreferredPropertyStar" : PreferredPropertyStar,
    "MaritalStatus" : MaritalStatus,
    "NumberOfTrips" : NumberOfTrips,
    "Passport" : Passport,
    "PitchSatisfactionScore" : PitchSatisfactionScore,
    "OwnCar" : OwnCar,
    "NumberOfChildrenVisiting" : NumberOfChildrenVisiting,
    "Designation" : Designation,
    "MonthlyIncome" : MonthlyIncome,
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction == 1 else "No"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
