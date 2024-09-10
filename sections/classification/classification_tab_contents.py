import streamlit as st
import pandas as pd
from sections.classification.ML_model import standardization_features, auto_ML_selection
from sections.classification.ml_model_choice import choice_split_dataset,choice_cut_dataframe,choice_standardization_features,choice_Balancing,choice_auto_ML_selection
from sections.classification.Analyse_df import colinearities, explicative_columns, load_and_encode,histogram_plot, pairplots,correlation_table
from sections.classification.cnn_model import dnn_modeling
from sections.classification.model_usage import predict_wine_type, download_prediction_files

def tab2_content(tab_visit,df_prep):
    list_choice2=[]
    st.write("# Analysis")
    st.write("A general overview of your datas")
    df = df_prep.copy()
    histogram_plot(df)
    # Identify columns of interest based on correlation with the target
    liste_col = list(df.columns[:-1])
    st.write("The correlation table between all your data")
    correlation_table(df, list_items=liste_col)  #

    explicative_columns(df)  # Return and print the correlation between target and feature with the thershold asked to the user

    list_choice = st.multiselect("Define your features of interest", options=liste_col, default=liste_col)
    # first_check = st.button("Afficher les corrélations")
    # first_check = st.text_input("Afficher les corrélations")
    if st.checkbox("Print correlations"):
        st.write("Pairplots for the selected features, dot colors correlate to wine category")
        pairplots(df, list_choice)
        colinearities(df, list_choice)

        list_choice2 = st.multiselect("Refine your features", options=liste_col, default=list_choice)
        df2 = standardization_features(df, list_choice2)
        tab_visit = "tab2"
        autoML = st.checkbox("Check for a fully automated analyses, else move to the next tab")

        if autoML:
            auto_ML_selection(df2)
        else:
            st.write("The autoanalysis will not be done")

    return tab_visit, liste_col, list_choice2, df

def tab3_content(tab_visit, liste_col, list_choice2,df):
    st.header("Play with parameters")
    if tab_visit == "tab2":
        list_choice3 = st.multiselect("Define the features of interest for your custom analyse", options=liste_col,
                                      default=list_choice2)
        df3 = choice_cut_dataframe(df, list_choice3)
        df3 = choice_standardization_features(df3, list_choice3)
        X_train, X_test, y_train, y_test = choice_split_dataset(df3)
        X_train, y_train = choice_Balancing(X_train, y_train)
        choice_auto_ML_selection(X_train, X_test, y_train, y_test)


    else:
        st.write(f"Start by running the analyse tab")

def tab4_content(tab_visit,liste_col,df):
    if tab_visit == "tab2":
        if st.checkbox("Launch the fully connected (dense) neural networks training"):
            dnn_modeling(df,liste_col)
    else:
        st.write(f"Start by running the analyse tab")

def tab5_content():
    predict_wine_type()
    download_prediction_files()