import streamlit as st
import streamlit_form_page as form
import streamlit_main as landing

def main():
    if "page" not in st.session_state:
        st.session_state.page = "form"

    if st.session_state.page == "form":
        form.form_page()  
    elif st.session_state.page == "streamlit_main":
        landing.landing_page()

if __name__ == "__main__":
    main()