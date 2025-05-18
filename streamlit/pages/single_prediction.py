# WiDS Datathon 2024 C2 - Part 3
# Streamlit Application
#
# Luna Jimenez Fernandez

# The goal of this file is to develop a MVP of an application used to deploy the
# trained model obtained through the full Data Science process.
#
# Since this app serves as an MVP using a simple model, an embedded system will be used.
# For real production and more complex models, a server - client model should be used,
# allowing access to the model via API
#
# This file serves as the main app entrypoint.
# The logic for both single prediction and batch prediction are contained in their separate files

# IMPORTS ##############################################################################################################

# Python
import pickle

# 3rd Party
import streamlit as st
import pandas as pd


# MODEL AND DATA LOADING ###############################################################################################
@st.cache_data
def load_model():
    """Loads the trained model from disk and un-pickles it"""
    with open("model.pkl", "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_data():
    """Loads the full training dataset as a DataFrame"""
    return pd.read_csv("data.csv", index_col="patient_id", dtype={"patient_zip3": object})


########################################################################################################################
# APP DEFINITION - MAIN PAGE ###########################################################################################

st.set_page_config(page_title="Single prediction")

# APP DEFINITION ######################################

# TITLES
# st.set_page_config(page_title="EMBEDDED - Estimador de problemas cardiacos")
# st.title("❤️ Estimador de problemas cardiacos 🏥")
# st.markdown("### ¿Cuanto riesgo tienes de sufrir un problema cardiaco?")
#
# # INPUT WIDGETS
# age = st.slider("Age", value=20, min_value=0, max_value=100)
# gender = st.radio("Gender", options=[
#     "Male",
#     "Female"
# ])
# rbp = st.number_input("Resting blood pressure",
#                       min_value=0)
# cholesterol = st.number_input("Cholesterol",
#                               min_value=0)
# fbs = st.radio("Fasting blood sugar", options=[
#                     "Yes",
#                     "No"
#                ],
#                index=1)
# mhr = st.number_input("Maximum heart rate",
#                       min_value=0)
#
# st.markdown("---")
#
# chosen_model = st.selectbox("Model to use for predictions", [
#     "Random Forest",
#     "Neural Network"
# ])
#
# # INPUT PROCESSING
# # Conversion dictionaries
# gender_dict = {"Male": 0, "Female": 1}
# fbs_dict = {"Yes": 1, "No": 0}
# outcome_dict = {
#     0: "💚 Probablemente no tendrás problemas de corazón",
#     1: "💔 Tienes riesgo de padecer problemas de corazón"
# }
#
# # Transform the input into a DataFrame for the models
# input = pd.DataFrame(
#     data=[[
#         age,
#         gender_dict[gender],
#         rbp,
#         cholesterol,
#         fbs_dict[fbs],
#         mhr
#     ]],
#     columns=["Age", "Sex", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"]
# )
#
# # MODEL AND DISPLAY
# # Pre-load both models
# rf_model = load_random_forest_model()
# nn_model = load_neural_network_model()
#
# # Depending on the chosen model, process the input
# if chosen_model == "Random Forest":
#     # RANDOM FOREST - Predict and extract the prediction
#     prediction = rf_model.predict(input)[0]
#
# elif chosen_model == "Neural Network":
#     # NEURAL NETWORK - Extract the prediction and transform it into an integer value
#     prediction = round(nn_model.predict(input)[0][0])
#
# # Output the prediction
# st.text("Pronóstico:")
# st.text(outcome_dict[prediction])

###########################################################