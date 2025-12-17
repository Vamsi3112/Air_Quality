import streamlit as st

st.write("🚀 App started successfully")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Test"])

if page == "Home":
    st.title("Home Page Works ✅")
else:
    st.title("Test Page Works ✅")

