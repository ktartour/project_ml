import streamlit as st
from sections.classification.Analyse_df import load_and_encode

def classification_page():

    st.write("# Data management")
    # Sidebar file Upload
    st.write('#### Select an file to upload.')
    uploaded_file = st.file_uploader('', type=['csv', 'txt'], accept_multiple_files=False, key="Zeyneb")

    st.write("DataFrame study")
    if uploaded_file == None:
        df = load_and_encode()
        df = df.drop(columns=["Unnamed: 0"])
    else:
        df = load_and_encode(uploaded_file)
        try:
            df = df.drop(columns=["Unnamed: 0"])
        except:
            None

    st.dataframe(df)
    st.write("Select columns for analysis")

    col_select = st.multiselect(
        "Choose your columns",
        df.columns,
    )
    for col in col_select:
        NA_presence = df[col].isna().sum()
        if NA_presence > 0:

            st.markdown(f"{col} :red[**has lacking value(s)**]")
        else:
            st.write(f"{col} **do not have lacking value**")

    return df
