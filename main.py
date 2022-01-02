# Import UI Library
import streamlit as st

import requests
import pandas as pd
import joblib

st.title("DMC Subscriber Predictor")
st.caption(body="Portuguese Banking Institution")


def run_prediction():
    # add the sidebar in the page
    st.sidebar.title("Prediction Information")
    option = st.sidebar.radio("Prediction Algorithm", options=['LR Model', 'RF Model', 'Gaussian NB'])
    # rf_model_checked = st.sidebar.checkbox("Predict using RF Algorithm", key='RF Model')

    left_col, center_col, right_col = st.columns(3)

    with left_col:
        age = st.number_input(label="Enter Age", min_value=10, max_value=150, step=1)
        education = st.selectbox(label="Education",
                                 options=['secondary', 'tertiary', 'unknown', 'primary'])
        housing = st.selectbox(label="Has housing loan?", options=['no', 'yes', 'unknown'])
        duration = st.slider(label="Enter Duration", min_value=1, max_value=5000, step=1)
        pdays = st.number_input(
            label="Number of days passed by after the client was last contacted",
            min_value=0, max_value=1000, value=999)
        predict = st.button(label="Predict")
    with center_col:
        marital = st.selectbox(label="Marital Status",
                               options=['married', 'single', 'unknown', 'divorced'])
        cc_default = st.selectbox(label="Has credit in default?", options=['no', 'yes', 'unknown'])
        loan = st.selectbox(label="Has personal loan?", options=['no', 'yes', 'unknown'])
        month = st.select_slider(label="Last contact month of year ",
                                 options=['jan', 'feb', 'mar', 'apr', 'may', 'june', 'jul', 'aug', 'sep',
                                          'oct', 'nov', 'dec'])
        previous = st.number_input(
            label="Number of contacts performed before this campaign and for this client ",
            min_value=0, max_value=1000, value=999)

    with right_col:
        job = st.selectbox(label="Job", options=['admin', 'blue-collar', 'entrepreneur', 'housemaid',
                                                 'management', 'retired', 'self-employed', 'services', 'student',
                                                 'technician', 'unemployed', 'unknown'])
        balance = st.number_input(label="Balance amount in the account", min_value=0.00, value=0.00, format='%.2f')
        contact = st.selectbox(label="Mode of Contact?", options=['cellular', 'telephone', 'unknown'])
        campaign = st.slider(label="Contacts during campaign for client", min_value=1,
                             max_value=50)
        poutcome = st.selectbox(label="Outcome of the previous marketing campaign ",
                                options=['failure', 'nonexistent', 'success'])
        day = st.slider(
            label="Day of the Month",
            min_value=1, max_value=31)

    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': cc_default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }

    if predict and option == 'LR Model':
        features = ['balance', 'day', 'month', 'duration', 'campaign', 'previous',
                    'default_no', 'housing_yes', 'housing_no', 'loan_no',
                    'contact_unknown']
        # response = requests.post("http://localhost:8000/predict", json=data)
        # prediction = response.text
        encoder = joblib.load("dump/encoding_pipeline.dump")
        model = joblib.load("dump/lr_model.dump")
        scale = joblib.load("dump/scale.dump")
        data = pd.DataFrame(pd.Series(data)).T
        data = encoder.transform(data)
        data = pd.DataFrame(scale.transform(data), columns=data.columns)
        predicted_val = model.predict(data[features])
        prob = model.predict_proba(data[features]).max()
        if predicted_val == 1:
            subscribe = 'Yes'
            delta_clr = 'normal'
        else:
            subscribe = 'No'
            delta_clr = 'inverse'
        st.sidebar.metric(label="Is User Going to Subscribe? ", value=subscribe,
                          delta=f"Probability: {round(prob, 2)}", delta_color=delta_clr)


    elif predict and option == 'RF Model':
        features = ['education', 'duration', 'poutcome', 'marital_married',
                    'marital_divorced', 'default_no', 'housing_yes', 'housing_no',
                    'loan_no', 'contact_unknown', 'contact_telephone']
        encoder = joblib.load("dump/encoding_pipeline.dump")
        model = joblib.load("dump/rf_model.dump")
        scale = joblib.load("dump/scale.dump")
        data = pd.DataFrame(pd.Series(data)).T
        data = encoder.transform(data)
        data = pd.DataFrame(scale.transform(data), columns=data.columns)
        predicted_val = model.predict(data[features])
        prob = model.predict_proba(data[features]).max()
        if predicted_val == 1:
            subscribe = 'Yes'
            delta_clr = 'normal'
        else:
            subscribe = 'No'
            delta_clr = 'inverse'
        st.sidebar.metric(label="Is User Going to Subscribe? ", value=subscribe,
                          delta=f"Probability: {round(prob, 2)}", delta_color=delta_clr)

    elif predict and option == 'Gaussian NB':
        features = ['education', 'duration', 'poutcome', 'marital_married',
                    'marital_divorced', 'default_no', 'housing_yes', 'housing_no',
                    'loan_no', 'contact_unknown', 'contact_telephone']
        encoder = joblib.load("dump/encoding_pipeline.dump")
        model = joblib.load("dump/gaussian.model")
        scale = joblib.load("dump/scale.dump")
        data = pd.DataFrame(pd.Series(data)).T
        data = encoder.transform(data)
        data = pd.DataFrame(scale.transform(data), columns=data.columns)
        predicted_val = model.predict(data[features])
        prob = model.predict_proba(data[features]).max()
        if predicted_val == 1:
            subscribe = 'Yes'
            delta_clr = 'normal'
        else:
            subscribe = 'No'
            delta_clr = 'inverse'
        st.sidebar.metric(label="Is User Going to Subscribe? ", value=subscribe,
                          delta=f"Probability: {round(prob, 2)}", delta_color=delta_clr)


if __name__ == "__main__":
    run_prediction()
