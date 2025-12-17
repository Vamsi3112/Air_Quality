import streamlit as st

st.write("🚀 App started successfully")

def main():
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", ["Home", "EDA", "Maps", "Model"])

    if page == "Home":
        st.title("Home Page Works ✅")
    elif page == "EDA":
        st.title("EDA Page")
    elif page == "Maps":
        st.title("Maps Page")
    elif page == "Model":
        st.title("Model Page")

if __name__ == "__main__":
    main()
