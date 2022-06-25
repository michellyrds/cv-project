import streamlit as st

from app.utils import sidebar_html


def __about_app__():
    st.markdown("Descrever nossa aplicação aqui")

    st.markdown(
        sidebar_html,
        unsafe_allow_html=True,
    )
