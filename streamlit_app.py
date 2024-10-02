import streamlit as st
import google.generativeai as genai

# Page Customisation
st.set_page_config(
   page_title="Cancer Diagnosis AI",
   page_icon="🧑‍⚕️",
)

# Show title and description.
st.title("🧑‍⚕️ Cancer Diagnosis AI")
st.write(
    "Upload a document below or ask a question – Gen AI will answer! "
    "To use this app, upload a document for cancer diagnosis or just ask anything related to cancer."
)

# Gemini API
genai.configure(api_key="AIzaSyDL4FArHVQaHu7ego_1188hyvZ6bZ018Uc")

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document"
)

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?"
)

if uploaded_file and question:
# Process the uploaded file and question.
    document = uploaded_file.read().decode()
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {question}",
        }
    ]

try:
    # Generate an answer using the Gemini API.
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(question)
    # Stream the response to the app using `st.write_stream`.
    st.write(response.text)
except TypeError:
    pass
