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
# This file serves as the logic for batch prediction of instances
# Due to its nature, the page is less user-interactive - mostly focused on uploading and downloading the results

# IMPORTS ##############################################################################################################

# Python
import pickle
from pathlib import Path
import json

# 3rd Party
import scipy as sp
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# MODEL AND DATA LOADING ###############################################################################################
@st.cache_data
def load_model():
    """Loads the trained model from disk and un-pickles it"""
    with open("model.pkl", "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def convert_for_download(df):
    """Given a Dataframe, prepares the CSV file to download"""
    return df.to_csv().encode("utf-8")


########################################################################################################################
# APP DEFINITION - SINGLE PREDICTION PAGE ##############################################################################

# TITLES ############################################
# Titles
st.set_page_config(page_title="Batch prediction", layout="wide")
st.title("Metastatic Cancer Diagnosis - Delay in diagnosis prediction")
st.markdown("""
#### Women in Data Science Datathon 2024  - Challenge 2 / Developed by Luna Jimenez Fernandez
## Batch prediction

""")

# Add a warning for first time loading
if "loaded" not in st.session_state:
    st.session_state.loaded = True
    st.warning("NOTE - Due to how Pickle and loading works, the model may give an error on first launch. Reloading"
               "the page should fix the problem.")

# FILE UPLOAD ##########################################################################################################
uploaded_file = st.file_uploader(
    label="Batch upload",
    type="csv",
    accept_multiple_files=False,
    help="Information about the patient must be uploaded as a CSV file "
         "where each row represents medical data about a patient"
)

# FILE PROCESSING ######################################################################################################

# Pre-load the model
model = load_model()

# Check if a file has been loaded
df_predictions = None
df_display = None

if uploaded_file is not None:
    # Read the CSV file - no need to input missing values, as that is processed by the model
    df_loaded = pd.read_csv(uploaded_file, index_col="patient_id", dtype={
                "patient_zip3": object,
                "breast_cancer_diagnosis_code": str,
                "metastatic_cancer_diagnosis_code": str
            })

    # Load the model and obtain the predictions for each patient
    # NOTE - Predictions are treated as done during model training
    # Predictions are rounded and converted into whole days

    df_predictions = pd.DataFrame(
            data=model.predict(df_loaded),
            columns=["metastatic_diagnosis_period"],
            index=df_loaded.index
        ).round(0).astype(int)

    # Compute a new DataFrame for display
    df_display = (
        df_predictions.join(
            df_loaded[["patient_age", "patient_race", "payer_type",
                       "breast_cancer_diagnosis_code", "metastatic_cancer_diagnosis_code",
                       "patient_state", "education_bachelors", "education_college_or_above",
                       "labor_force_participation", "family_dual_income"]],
            on="patient_id"
        )
    )

    st.success("File processed successfully")

# RESULTS DISPLAY ######################################################################################################


# Display the DataFrame - if it exists
if df_display is not None:

    st.dataframe(df_display,
                 column_config={
                     "_index": "Patient ID",
                     "metastatic_diagnosis_period": "Predicted diagnosis period",
                     "patient_age": "Age",
                     "patient_race": "Race",
                     "payer_type": "Healthcare",
                     "breast_cancer_diagnosis_code": "Breast cancer diagnosis",
                     "metastatic_cancer_diagnosis_code": "Metastatic cancer diagnosis",
                     "patient_state": "State",
                     "education_bachelors": "Bachelors (%)",
                     "education_college_or_above": "College studies (%)",
                     "labor_force_participation": "Labor force (%)",
                     "family_dual_income": "Dual-income families (%)"
                 },
                 column_order=["patient_age", "patient_race", "payer_type",
                               "breast_cancer_diagnosis_code", "metastatic_cancer_diagnosis_code",
                               "patient_state", "education_bachelors", "education_college_or_above",
                               "labor_force_participation", "family_dual_income",
                               "metastatic_diagnosis_period"])

    download_data = convert_for_download(df_predictions)

    st.download_button(
        label="Download predictions",
        data=download_data,
        file_name="processed_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore"
    )

