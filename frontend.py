import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Motherboard Health AI",
    layout="centered"
)

st.title("💻 Motherboard Health AI")
st.write("Enter the motherboard parameters and click **Predict**.")

with st.form("prediction_form"):

    model_name = st.text_input("Model Name", "Dell Inspiron 6880")

    cpu = st.number_input(
        "CPU Usage (%)",
        min_value=0.0,
        max_value=100.0,
        value=82.0
    )

    ram = st.number_input(
        "RAM Usage (%)",
        min_value=0.0,
        max_value=100.0,
        value=76.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        value=91.0
    )

    voltage = st.number_input(
        "Voltage (V)",
        min_value=0.0,
        value=10.2
    )

    disk = st.number_input(
        "Disk Usage (%)",
        min_value=0.0,
        max_value=100.0,
        value=67.0
    )

    fan = st.number_input(
        "Fan Speed (RPM)",
        min_value=0,
        value=1450
    )

    submit = st.form_submit_button("Predict")

if submit:

    payload = {
        "ModelName": model_name,
        "CPUUsage": cpu,
        "RAMUsage": ram,
        "Temperature": temperature,
        "Voltage": voltage,
        "DiskUsage": disk,
        "FanSpeed": fan
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction Complete")

            st.subheader("Prediction")

            for key, value in result.items():

                if isinstance(value, dict):

                    st.markdown(f"### {key.replace('_', ' ').title()}")

                    for k, v in value.items():

                        if isinstance(v, list):

                            st.write(f"**{k.replace('_',' ').title()}**")

                            for item in v:
                                st.write(f"• {item}")

                        else:
                            st.write(f"**{k.replace('_',' ').title()} :** {v}")

                elif isinstance(value, list):

                    st.markdown(f"### {key.replace('_',' ').title()}")

                    for item in value:
                        st.write(f"• {item}")

                else:

                    st.write(
                        f"**{key.replace('_',' ').title()} :** {value}"
                    )

        else:

            st.error(f"API Error : {response.text}")

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI backend.\n\n"
            "Run:\n\n"
            "uvicorn app:app --reload"
        )