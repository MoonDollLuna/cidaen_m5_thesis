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

import streamlit as st


# MODEL METHODS - TO AVOID DUE TO PICKLING ERRORS ######################################################################
def to_categorical(dataframe):
    """
    Transforms all the elements in a DataFrame into Categorical dTypes.

    Used instead of a lambda function for pickling purposes
    """

    return dataframe.astype("category")


########################################################################################################################
# APP DEFINITION - MAIN PAGE ###########################################################################################

# Create the navigation side-bar
pages = {
    "Metastatic diagnosis prediction": [
        st.Page("pages/single_prediction.py", title="Patient prediction", default=True),
        st.Page("pages/batch_prediction.py", title="Batch prediction")
    ]
}

# Load the pages and run the information
pg = st.navigation(pages)
pg.run()

