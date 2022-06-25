import streamlit as st

from app.utils import load_images, sidebar_html


def __about_app__():
    st.markdown("# Ex Machina, a IA de detectar faces")
    load_images("media/banner.png", width=704)
    st.markdown(
        "#### Aplicação de visão computacional desenvolvida para o 7º semestre"
        " do curso de Sistemas de Informação - EACH USP."
    )

    st.markdown(
        """
        **Projeto desenvolvido por:**\n
        Leonardo Nogueira Cordeiro  
        Mateus Alex dos Santos Luna  
        Michelly Rodrigues da Silva  
        Pedro Gabriel dos Anjos Santana  
        Raul Sperandio  
        Vitor Hugo Brasiliense da Rosa
        """
    )

    st.markdown(
        sidebar_html,
        unsafe_allow_html=True,
    )
