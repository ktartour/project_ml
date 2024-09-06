from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE

def standardization_features(df, list_columns):
    df2= pd.DataFrame()
# standardization des colonnes de list_to_norm
    for col in list_columns:
        df2[col] = (df[col] - df[col].mean()) / df[col].std()

    df2["target"]=df["target"]
    return df2

def split_dataset(df,test_size=0.2):
#split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
def Balancing(X_train, y_train):

    categories_counts = y_train.value_counts()
    categories_counts = categories_counts.rename({0: "Vin sucré", 1: "Vin éuilibré", 2: "Vin amer"})
    categories_counts = categories_counts.to_frame()
    categories_counts["percent"] = 100 * round(categories_counts["count"] / categories_counts["count"].sum(), 4)

    st.write(categories_counts)  ##

    if ((categories_counts["percent"] > 43).any()) or ((categories_counts["percent"] < 25).any()):
        st.write("The dataset is imbalanced, it will be corrected with SMOTE oversampling")
        smote = SMOTE()  # Oversampling method
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        return X_train_smote, y_train_smote
    else:
        st.write("The dataset is considered balanced")

        return X_train, y_train

def auto_ML_selection(df2):
# Define the parameter grid to search over
    X_train, X_test, y_train, y_test = split_dataset(df2)
    X_train, y_train = Balancing(X_train, y_train)


    param_grid = {'LogisticRegression':{
        'max_iter':[10000, 1000, 100, 10], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
        'DecisionTreeClassifier':{
        'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
        'RandomForestClassifier':{
        'n_estimators': [100,5000,10000], 'max_depth': [None, 20, 30]},
        'KNeighborsClassifier':{
        'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'SVC':{
        'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }
# models to test: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), Naive Bayes, K-Nearest Neighbors (KNN)
    models = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(),
        KNeighborsClassifier()
    ]



    for model in models:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model.__class__.__name__], cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best parameters and the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        #Print the classification report
        report_dict = classification_report(y_test, y_pred,output_dict=True)  # Set output_dict=True to return a dictionary
        df_classification_report = pd.DataFrame(report_dict)
        st.write(f"The model {model.__class__.__name__} has the following performance:")
        st.write(df_classification_report)
        st.write(f"The best parameters for the {model.__class__.__name__} are {best_params}")

        with st.expander("See confusion matrix"):

            # Create a new figure for the confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title(f"Confusion Matrix of {model.__class__.__name__}")

            # Show the plot in Streamlit
            st.pyplot(fig)  # Now we pass the created figure object to Streamlit

        #save the model as a binary avec scikit learn
        # save the model to disk
        #filename = f'sample_data/{model.__class__.__name__}.sav'
        #pickle.dump(model, open(filename, 'wb'))