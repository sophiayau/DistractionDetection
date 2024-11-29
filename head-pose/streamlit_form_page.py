"""
testing streamlit UI for our front end
"""

import streamlit as st
from streamlit_navigation_bar import st_navbar
from datetime import date, timedelta
from streamlit_main import landing_page

def form_page():

    if "page" not in st.session_state:
        st.session_state.page = "form" 

    st.markdown("<h1 style='text-align: center; color: black;'>Focus Point</h1>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    name = col1.text_input("Name")
    birthday = col2.date_input("Date of Birth (YYYY/MM/DD)", min_value=date(1900, 1, 1), max_value=(date.today()))

    gender_options = ["Female", "Male", "Other"]
    gender = col3.selectbox("Gender", gender_options)
    occupation = col4.text_input("What is your occupation?")

    options = ["I want to increase my productivity", "I'm looking to monitor my attention span", "I'm trying to break my Netflix binge cycle", "Other"]
    dropdown = st.selectbox("What made you want to use our application?", options)


    email_option = st.radio("Would you like your results emailed to you?", ["Yes", "No, ignorance is bliss"])

    if email_option == "Yes":
        email_input = st.text_input("Please enter your email address")

    st.write("")

    if st.button("Let's get started!"):
        st.session_state.name = name
        st.session_state.birthday = birthday
        st.session_state.gender = gender
        st.session_state.occupation = occupation
        if email_option == "Yes":
            st.session_state.email = email_input
        st.session_state.page = "streamlit_main"

    st.divider()


