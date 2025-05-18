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
from pathlib import Path

# 3rd Party
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import streamlit as st


# MODEL AND DATA LOADING ###############################################################################################
@st.cache_data
def load_model():
    """Loads the trained model from disk and un-pickles it"""
    with open("model.pkl", "rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_data():
    """Loads the full training dataset as a DataFrame"""
    return (
        pd.read_csv("data.csv", index_col="patient_id", dtype={"patient_zip3": object})
        .fillna({
            "payer_type": "UNKNOWN",
            "patient_race": "UNKNOWN"
        })
    )


@st.cache_data
def load_icd10_data():
    """Loads the full ICD10 medical dataset from a json file"""
    return pd.read_json("codes/icd10_codes.json").set_index("code")


@st.cache_data
def load_icd9_data():
    """Loads the full ICD9 medical dataset from a CSV file"""
    # Read the dataframe and remove the dots
    df = pd.read_csv("codes/icd9_codes.csv")
    df["icd9code"] = df["icd9code"].str.replace(".", "")

    return df.set_index("icd9code")["long_description"]


# CACHED HELPER METHODS ################################################################################################
@st.cache_data
def get_icd_description(icd_code):
    """
    Given an ICD code, return the appropriate description.

    If the code is not directly contained in the dataset, find the closest code
    """
    return (f"{icd_code} - {load_icd10_data().filter(like=icd_code, axis=0).iloc[0]['desc']}" if icd_code[0] == "C"
                     else f"{icd_code} - {load_icd9_data().filter(regex=f'^{icd_code}', axis=0).iloc[0]}")


@st.cache_data
def get_average_percentage_state(data, state, attribute):
    """
    Given an attribute, computes the average score for the specified state

    Used to automatically fill in values when choosing a state
    """

    return (
        data[data["patient_state"]==state][attribute].mean()
    )


########################################################################################################################
# APP DEFINITION - SINGLE PREDICTION PAGE ##############################################################################

# TITLES ############################################
# Titles
st.set_page_config(page_title="Patient prediction", layout="wide")
st.title("Metastatic Cancer Diagnosis - Delay in diagnosis prediction")
st.markdown("""
## Patient prediction
#### Women in Data Science 2024 - Challenge 2
""")

# Prepare session state values
if "reloaded" not in st.session_state:
    st.session_state.reloaded = False

# FILE UPLOAD AND PRE-PROCESSING ####################

# Storage of the patient info
patient_info = None

uploaded_file = st.file_uploader(
    label="Quick upload",
    type=["csv", "json"],
    accept_multiple_files=False,
    help="Information about the patient can be uploaded as either a CSV table or a JSON file"
)

# Check if a file has been loaded

if uploaded_file is not None:

    # Files will always be read as Pandas DataFrames
    df = None
    # Check the format of the file and load it accordingly
    extension = Path(uploaded_file.name).suffix

    # CSV files
    if extension == ".csv":
        df = (
            pd.read_csv(uploaded_file, index_col="patient_id", dtype={
                "patient_zip3": object,
                "breast_cancer_diagnosis_code": str,
                "metastatic_cancer_diagnosis_code": str
            })
            .fillna({
                "payer_type": "UNKNOWN",
                "patient_race": "UNKNOWN"
            })
        )

    # JSON files
    elif extension == ".json":
        df = (
            pd.read_json(uploaded_file, typ="series", dtype={
                "patient_zip3": object,
                "breast_cancer_diagnosis_code": str,
                "metastatic_cancer_diagnosis_code": str
            })
            .to_frame().transpose()
            .fillna({
                "payer_type": "UNKNOWN",
                "patient_race": "UNKNOWN"
            })
        )

    # Check if the CSV contains more than one instance - and extract the first one
    if len(df) > 1:
        st.warning("""
            More than one patient contained within the file - only the first one will be considered.
            Did you mean to do a batch prediction?
            """)

    patient_info = df.iloc[0]
    # Sanity check - ensure that the diagnosis codes are ALWAYS treated as strings
    patient_info["breast_cancer_diagnosis_code"] = str(patient_info["breast_cancer_diagnosis_code"])
    patient_info["metastatic_cancer_diagnosis_code"] = str(patient_info["metastatic_cancer_diagnosis_code"])

st.divider()

# PATIENT INFORMATION ##################################################################################################
# If a patient has been loaded, automatically load the information into the form

# Pre-load the Dataframe for reuse
df_data = load_data()

# Store the patient form submission in a dictionary
patient_submission = {}

st.subheader("Patient information:")

with st.form("single_form"):
    # PART 1 - Medical data
    # Age, race and payer type
    age_col, race_col, payertype_col = st.columns(3)
    with age_col:
        patient_submission["patient_age"] = st.number_input(
            label="Patient age",
            min_value=0,
            max_value=150,
            value=patient_info["patient_age"] if patient_info is not None else 50
        )

    with race_col:
        patient_submission["patient_race"] = st.selectbox(
            label="Patient race",
            options=df_data["patient_race"].unique(),
            index=(
                np.where(df_data["patient_race"].unique() == patient_info["patient_race"])[0].item()
                if patient_info is not None else 0
            )
        )

    with payertype_col:
        patient_submission["payer_type"] = st.selectbox(
            label="Patient healthcare type",
            options=df_data["payer_type"].unique(),
            index=(
                np.where(df_data["payer_type"].unique() == patient_info["payer_type"])[0].item()
                if patient_info is not None else 0
            )
        )

    code_col1, code_col2 = st.columns(2)
    with code_col1:
        ordered_breast_cancer_codes = np.sort(df_data["breast_cancer_diagnosis_code"].unique())[::-1]
        patient_submission["breast_cancer_diagnosis_code"] = st.selectbox(
            label="Breast cancer diagnosis code",
            options=ordered_breast_cancer_codes,
            format_func=get_icd_description,
            index=(
                np.where(ordered_breast_cancer_codes == patient_info["breast_cancer_diagnosis_code"])[0].item()
                if patient_info is not None else 0
            )
        )

    with code_col2:
        ordered_metastatic_cancer_codes = np.sort(df_data["metastatic_cancer_diagnosis_code"].unique())[::-1]
        patient_submission["metastatic_cancer_diagnosis_code"] = st.selectbox(
            label="Metastatic cancer diagnosis code",
            options=ordered_metastatic_cancer_codes,
            format_func=get_icd_description,
            index=(
                np.where(ordered_metastatic_cancer_codes == patient_info["metastatic_cancer_diagnosis_code"])[0].item()
                if patient_info is not None else 0
            )
        )

    st.divider()

    # PART 2 - Socio-economic data
    # State selection
    patient_submission["patient_state"] = st.selectbox(
        label="Patient state",
        options=df_data["patient_state"].unique(),
        index=(
            np.where(df_data["patient_state"].unique() == patient_info["patient_state"])[0].item()
            if patient_info is not None else 0
        )
    )

    # Percentages
    # If no data is loaded, the default value of a percentage is whatever the state is
    reload_percentages = st.form_submit_button("Compute percentages for chosen state")
    st.markdown("#### Percentage of the population of the patient state that:")


    # If the state has been reloaded, display a message and reset the state
    if st.session_state.reloaded:
        st.success("Percentages automatically computed for the current state")

    percent_col1, percent_col2 = st.columns(2)
    with percent_col1:
        patient_submission["education_bachelors"] = st.number_input(
            label="Has a bachelors degree",
            min_value=0.0,
            max_value=100.0,
            value=(
                get_average_percentage_state(
                    df_data,
                    patient_submission["patient_state"],
                    "education_bachelors"
                ) if st.session_state.reloaded
                else patient_info["education_bachelors"] if patient_info is not None
                else 50.0
            )
        )

        patient_submission["labor_force_participation"] = st.number_input(
            label="Is part of the labor force",
            min_value=0.0,
            max_value=100.0,
            value=(
                get_average_percentage_state(
                    df_data,
                    patient_submission["patient_state"],
                    "labor_force_participation"
                ) if st.session_state.reloaded
                else patient_info["labor_force_participation"] if patient_info is not None
                else 50.0
            )
        )

    with percent_col2:
        patient_submission["education_college_or_above"] = st.number_input(
            label="Has studied in college",
            min_value=0.0,
            max_value=100.0,
            value=(
                get_average_percentage_state(
                    df_data,
                    patient_submission["patient_state"],
                    "education_college_or_above"
                ) if st.session_state.reloaded
                else patient_info["education_college_or_above"] if patient_info is not None
                else 50.0
            )
        )

        patient_submission["family_dual_income"] = st.number_input(
            label="Is part of a dual-income household",
            min_value=0.0,
            max_value=100.0,
            value=(
                get_average_percentage_state(
                    df_data,
                    patient_submission["patient_state"],
                    "family_dual_income"
                ) if st.session_state.reloaded
                else patient_info["family_dual_income"] if patient_info is not None
                else 50.0
            )
        )

    # UPLOAD BUTTON
    submitted = st.form_submit_button("Predict")

    if reload_percentages:
        st.session_state.reloaded = True
        st.rerun()

########################################################################################################################
# MODEL OUTPUT #########################################################################################################

# Load the model from disk
# NOTE - A function needs to be loaded in the Main part of the app due to pickle serialization problems
model = load_model()

# Transform the dictionary into a DataFrame to feed into the model
df_input = pd.DataFrame.from_records(patient_submission, index=["patient"])

# Obtain the prediction for the input
prediction = model.predict(df_input)
processed_prediction = int(round(prediction.item(), 0))

st.subheader("Predicted metastatic diagnosis period:")
st.metric("",
          value=f"{processed_prediction} days",
          label_visibility="collapsed")





