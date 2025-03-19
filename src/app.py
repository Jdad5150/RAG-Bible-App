import streamlit as st
from inference import load_db, get_query, get_cohere_response, retrieval_model
import numpy as np
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Cache expensive operations
@st.cache_data
def load_data():
    index, metadata = load_db()
    df = pd.read_csv('data/kjv_cleaned.csv')
    return index, metadata, df

index, metadata, df = load_data()

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state.generated = None  # Stores OpenAI response
if 'retrieved_context' not in st.session_state:
    st.session_state.retrieved_context = None  # Stores retrieved passages
if 'user_query' not in st.session_state:
    st.session_state.user_query = None  # Stores user query

# Function to send email
def send_email(user_query, retrieved_context, cohere_response, user_name , user_feedback):
    sender_email = "jjvlittle@gmail.com" 
    receiver_email = "jjvlittle@gmail.com" 
    password = "zaij hysy scva vfos"

    # Create the email
    subject = "User Feedback from Bible Quest"
    body = f"""
    <html>
    <body>
        <h2 style="color: #4CAF50;">User Feedback from Bible Search Engine</h2>
        
        <h3 style="color: #333;">User Details</h3>
        <p><strong>Name:</strong> {user_name}</p>
        
        <h3 style="color: #333;">User Query</h3>
        <p style="background-color: #f9f9f9; padding: 10px; border-radius: 5px;">{user_query}</p>
        
        <h3 style="color: #333;">Retrieved Context</h3>
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px;">
            {retrieved_context.replace("\n", "<br>")}
        </div>
        
        <h3 style="color: #333;">AI Response</h3>
        <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px;">
            {cohere_response.replace("\n", "<br>")}
        </div>
        
        <h3 style="color: #333;">User Feedback</h3>
        <p style="background-color: #f9f9f9; padding: 10px; border-radius: 5px;">{user_feedback}</p>
    </body>
    </html>
    """

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "html"))

    # Send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        st.success("Thank you for your feedback! It has been sent successfully.")
    except Exception as e:
        st.error(f"Failed to send feedback. Error: {e}")

# App title and description
st.markdown(
    "<h1 style='text-align: center; color: blue;'>Bible Quest 🕊️</h1>",
    unsafe_allow_html=True
)

header_container = st.container(border=True)
header_container.warning("This is a prototype version of the application. This is not representative of how the UI will look. The final version will be more polished without bugs.")
header_container.markdown("### Enter a query (e.g., 'What does the Bible say about love?') to retrieve relevant passages and get an AI-generated response.")




# User input
user_query = st.text_input('Enter a query:', key='input')

if user_query.strip() == "":
    st.warning("Please enter a query.")
    st.stop()

# Process the query
if user_query:
    with st.spinner('Searching the Bible and generating a response...'):
        # Encode the query and search the index
        query_embedding = retrieval_model.encode(user_query)
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)

        k = 5
        distance, indices = index.search(query_embedding, k)

        # Format retrieved passages
        retrieved_context = "\n\n".join(
            [f"**{metadata[idx]['Book Name']} {metadata[idx]['Chapter']}:{metadata[idx]['Verse']}**\n"
            f"{df.iloc[idx]['Text']}"
            for idx in indices[0]]
        )
        st.session_state.retrieved_context = retrieved_context
        st.session_state.user_query = user_query

        # Generate OpenAI response
        try:
            answer = get_cohere_response(retrieved_context, user_query)
            st.session_state.generated = answer
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Display OpenAI response
if st.session_state.generated:
    st.subheader("Response:")
    st.write(st.session_state.generated)

# Collapsible section for retrieved context
if st.session_state.retrieved_context:
    with st.expander("View Retrieved Context"):
        st.markdown(st.session_state.retrieved_context)

# Feedback form
if st.session_state.generated:
    st.subheader("Feedback")
    with st.form("feedback_form"):
        st.text("Please provide feedback on the response:")

        first_name = st.text_input("First Name:")
        last_name = st.text_input("Last Name:")
        user_feedback = st.text_area("Please provide feedback on the response:")

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if user_feedback.strip() == "":
                st.warning("Please provide feedback before submitting.")
            else:
                send_email(
                    st.session_state.user_query,
                    st.session_state.retrieved_context,
                    st.session_state.generated,
                    user_name = first_name + " " + last_name,
                    user_feedback=user_feedback
                )

                first_name = ""
                last_name = ""
                user_feedback = ""

# Clear results button
if st.button('Clear Results'):
    st.session_state.generated = None
    st.session_state.retrieved_context = None
    st.session_state.user_query = None