import streamlit as st
from sections.classification.Analyse_df import load_and_encode
def classification_page():

    st.header("Data management")
    # Sidebar file Upload
    st.write('#### Select an file to upload.')
    uploaded_file = st.file_uploader('', type=['csv', 'txt'], accept_multiple_files=False)

    st.write("Etudes DataFrames")
    df = load_and_encode()
    st.dataframe(df)
    st.write("Listes des colonnes")
    col_select = st.multiselect(
        "Choisissez vos colonnes",
        df.columns,
    )
    st.write("Valeurs Manquantes")
    df_somme_na = df.isna().sum()
    if sum(df_somme_na) > 0:
        st.write("Valeurs manquantes")
        st.write(df_somme_na)
    else:
        st.write("aucunes valeurs!")