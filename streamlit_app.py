import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data

def cell_data():
  st.sidebar.header("Cell Nuclei Measurements :")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

def add_predictions(input_data):
  model = pickle.load(open("model/model.pkl", "rb"))
  scaler = pickle.load(open("model/scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
  
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    # Page Customisation
    st.set_page_config(
        page_title="Cancer Predictor AI",
        page_icon="🧑‍⚕️",
    )
    with open("assets/style.css") as f:
      st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # Show title and description.
    st.title("🧑‍⚕️ Cancer Predictor AI")
    st.write(
        "Ask a question – Gen AI will answer! "
    )
    st.write(
        "To use this app, Type Cytology lab data for cancer diagnosis or just ask anything related to cancer."
    )
    
    # Gemini API
    genai.configure(api_key="AIzaSyDL4FArHVQaHu7ego_1188hyvZ6bZ018Uc")

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about cancer or type - Take Cytology Lab Data!",
        placeholder="Take my Cytology Lab Data."
    )
    
    # Whether the question containes taking data?
    a = "Take"
    b = "Cytology"

    if (a in question):
        if (b in question):
            input_data = cell_data()
            with st.container():
              add_predictions(input_data)

    else:
        try:
            # Generate an answer using the Gemini API.
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(question)
            # Stream the response to the app using `st.write_stream`.
            st.write(response.text)
        except TypeError:
            pass

if __name__ == '__main__':
    main()
