import streamlit as st
from sections.nailsdetection.nails import side_bar_nails, initalize_variables, print_info, treatment_nails
from sections.regression.regression import regression_page
from sections.classification.classification_tab_contents import tab2_content, tab3_content,tab4_content
from sections.classification.Preparation_df import classification_page

st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

type_data = st.sidebar.radio(
    "Choisissez votre type de playground",
    ["Regression", "Classification", "NailsDetection"]
)

if type_data == "Regression":
    regression_page()
elif type_data == "Classification":
    tab_visit = "None"
    tab1, tab2, tab3, tab4 = st.tabs(["Preparation", "Analyse", "play with parameters","Try a CNN"])
    with tab1:

        classification_page()
    with tab2:
        tab_visit, liste_col, list_choice2, df = tab2_content(tab_visit)
    with tab3:
        tab3_content(tab_visit, liste_col, list_choice2, df)
    with tab4:
        st.header("Try a CNN")
        tab4_content(tab_visit,list_choice2, df)

elif type_data == "NailsDetection":
    uploaded_file,confidence_threshold = side_bar_nails()
    Prediction, PolygonPoints, nails = initalize_variables()
    print_info()
    treatment_nails(uploaded_file, confidence_threshold,Prediction,PolygonPoints)


else:
    st.write("Choisissez une option")