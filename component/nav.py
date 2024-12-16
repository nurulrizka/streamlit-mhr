import streamlit as st

def navbar():
    # Membuat 4 kolom
    col1, col2, col3, col4 = st.columns(4, gap="small")

    # Menambahkan konten di dalam kolom pertama
    with col1:
        st.page_link("app.py", label="Home")
    with col2:
        st.page_link("pages/modelling.py", label="Modelling")
    with col3:
        st.page_link("pages/PreProcessing.py", label="Pre Processing")
    with col4:
        st.page_link("pages/dataUnderstanding.py", label="Data Understanding")



    
    