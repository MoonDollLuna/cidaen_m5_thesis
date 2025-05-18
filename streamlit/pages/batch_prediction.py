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


st.set_page_config(page_title="Batch prediction")