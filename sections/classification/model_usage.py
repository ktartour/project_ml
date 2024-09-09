import streamlit as st
import pandas as pd
import pickle
import glob
import ast

#Download a Excelfile with columns alcohol,malic_acid,ash,alcalinity_of_ash	magnesium,total_phenols	flavanoids,nonflavanoid_phenols,proanthocyanins,
#       color_intensity,hue,od280/od315_of_diluted_wines,proline

def predict_wine_type():

    files = glob.glob('modele/*.sav')
    model_to_use = st.radio(
        "Choose the model to apply to your datas",
        files
    )
    features_to_have = model_to_use.replace(".sav", "_features.txt")
    with open(features_to_have, "r") as file:
        content = file.read()
        st.write(f"The excel file should have at least the columns {content}")
        content = ast.literal_eval(content)


    uploaded_file = st.file_uploader('Select a csv file for wine analysis', type=['csv'], accept_multiple_files=False)
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write("The downloaded data are the following:")
        st.write(dataframe.head(5))
        i=60
        for col in content:
            NA_presence = dataframe[col].isna().sum()
            if NA_presence > 0:
                #mean median 0

                st.markdown(f"The column {col} :red[**has lacking value(s)**], they have to be replaced")
                type_correction = st.radio("How to replace NA?", ["By the mediane","By the mean","By 0.0"], key=i)
                if type_correction == "By the mediane":
                    dataframe[col] = dataframe[col].fillna(dataframe[col].median())
                elif type_correction == "By the mean":
                    dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
                elif type_correction == "By 0.0":
                    dataframe[col] = dataframe[col].fillna(0)
                i+=1
            else:
                st.write(f"The column {col} **do not have lacking value**")



        dataframe = dataframe[content]


        loaded_model = pickle.load(open(f"{model_to_use}", 'rb'))
        y_pred = loaded_model.predict(dataframe)
        dataframe["ypred"] = y_pred
        dataframe["ypred"] = dataframe["ypred"].replace(1, "Vin équilibré")
        dataframe["ypred"] = dataframe["ypred"].replace(0, "Vin amer")
        dataframe["ypred"] = dataframe["ypred"].replace(2, "Vin sucré")
        st.write("The predictions of wine categories follow")
        st.write(dataframe.head(5))
        filename_analyse = st.text_input("Filename for the estimation of wine categories (no space admitted)")
        if st.button("Click to validate the filename", key=50):
            # Save DataFrame to CSV
            dataframe.to_csv(f"Wine_categorisation/{filename_analyse}.csv", index=False)
            st.write("Estimation saved")
        else:
            st.write("Estimation not saved")