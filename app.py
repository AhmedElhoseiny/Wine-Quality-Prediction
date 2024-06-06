import pickle
import pandas as pd
import streamlit as st

# Load models and scaler
with open('gnb_model.pkl', 'rb') as f:
    gnb_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('inputs.pkl', 'rb') as f:
    inputs = pickle.load(f)

def prediction(model, input_data):
    df = pd.DataFrame([input_data], columns=inputs)
    df_scaled = scaler.transform(df)
    result = model.predict(df_scaled)[0]
    return result

def main():
    st.title("Wine Quality Classification")

    # Create input fields for each feature
    input_data = []
    for feature in inputs:
        value = st.number_input(f'Enter {feature}', value=0.0)
        input_data.append(value)
    
    # Display contact information in sidebar
    linkedin_url = "https://www.linkedin.com/in/ahmed-elhoseiny-2a952122a"
    github_url = "https://github.com/AhmedElhoseiny"
    email = "ahmedelhoseiny20022010@gmail.com"
    st.sidebar.image("Ahmed.jpg", width=100)
    st.sidebar.write("Connect with me:")
    st.sidebar.markdown(f"[![Email](https://img.shields.io/badge/Email-Contact-informational)](mailto:{email})")
    st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/GitHub-Profile-green)]({github_url})")
    st.sidebar.markdown(f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)]({linkedin_url})")

    if st.button("Predict with GaussianNB"):
        result = prediction(gnb_model, input_data)
        st.text("Prediction: " + ("Good Quality" if result == 1 else "Bad Quality"))
    
    if st.button("Predict with RandomForest"):
        result = prediction(rf_model, input_data)
        st.text("Prediction: " + ("Good Quality" if result == 1 else "Bad Quality"))

if __name__ == "__main__":
    main()
