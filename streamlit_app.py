import streamlit as st
import requests
import json

BACKEND_URL = "http://localhost:8000/chat/"

@st.cache_data
def get_insight_summary(insight_path):
    with open(insight_path, 'r') as file:
        payload = {'incoming_insight':json.load(file)}

    response = requests.post(url='http://127.0.0.1:8000/get_detailed_summary_of_clustered_incidents_for_chat', json=payload)
    summary_text = response.json()['summary_of_incidents']
    return summary_text

summary_text = get_insight_summary(insight_path = "./data/Incoming_Incident_Data/INC17829485.json")

st.session_state.messages = []

def send_message(user_message: str):
    payload = {"insight_summary": summary_text, "message" : user_message}
    response = requests.post(BACKEND_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("response")
    else:
        return f"Error: {response.status_code} - {response.text}"

def display_conversation():
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"**User**: {msg['message']}")
        else:
            st.markdown(f"**Assistant**: {msg['message']}")

st.title("Chat with Assistant")

user_input = st.text_input("Your message:")

if user_input:
    bot_response = send_message(user_input)
    st.session_state.messages.append({'role':'assistant', 'message':bot_response})

display_conversation()


