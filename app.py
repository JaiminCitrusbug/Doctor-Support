# app_chat.py 
import streamlit as st
from rag_chat import generate_answer

st.set_page_config(page_title="Doctor's RAG Chat Assistant", layout="centered")

st.title("🩺 Doctor's Assistant")
st.markdown("Chat with your AI medical assistant (RAG-powered).")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

# Chat input box
if prompt := st.chat_input("Type your question..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display it immediately
    with st.chat_message("user"):
        st.write(prompt)

    # Generate RAG answer with full context
    with st.chat_message("assistant"):
        # create a stable placeholder (no interim text) and update it in-place to avoid the skeletal/ghost text
        placeholder = st.empty()           # DO NOT write anything here (no placeholder.text/markdown)

        # show the spinner only (this will be visible while the model runs)
        with st.spinner("Analyzing with medical knowledge base..."):
            response = generate_answer(prompt, st.session_state.messages)

        # replace the empty placeholder with the final response (single in-place update, no duplicate)
        placeholder.write(response)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response})

