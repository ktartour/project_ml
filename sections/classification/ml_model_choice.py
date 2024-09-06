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

def choice_cut_dataframe(df, list_columns):
    list_col= list_columns +["target"]
    df2 = df[list_col]
    return df2
def choice_standardization_features(df, list_columns):
    df2 = pd.DataFrame()

    if st.checkbox("Standardize some columns?"):
        if st.checkbox("Standardize all columns?"):
            for col in list_columns:
               df2[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            list_col_to_std = st.multiselect("Define the features to standardize", options=list_columns,
                                              default=list_columns)
            for col in list_col_to_std:
                df2[col] = (df[col] - df[col].mean()) / df[col].std()
    else:
        df2 = df.copy()
    df2["target"]=df["target"]
    return df2

def choice_split_dataset(df):
#split the dataset
    l = 10
    t = 0
    while t == 0:
        try:
            size_train = float(size_train)
            if size_train<0.1 or size_train>0.9:
                t = 0
                size_train = st.text_input(
                    "which proportion of the dataset do you want to use as to train the model (between 0.1 and 0.9)",
                    "0.8", key=l)
                l += 1
            else:
                t = 1
        except:
            size_train = st.text_input(
                "which proportion of the dataset do you want to use as to train the model (between 0.1 and 0.9)", "0.8", key=l)
            l += 1

    test_size = 1 - size_train

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
def choice_Balancing(X_train, y_train):

    categories_counts = y_train.value_counts()
    categories_counts = categories_counts.rename({0: "Vin sucré", 1: "Vin éuilibré", 2: "Vin amer"})
    categories_counts = categories_counts.to_frame()
    categories_counts["percent"] = 100 * round(categories_counts["count"] / categories_counts["count"].sum(), 4)
    st.write("Count and repartition of targets")
    st.write(categories_counts)

    if st.checkbox("Balancing groups by over sampling (SMOTE), usually when at least 1/3 population is less than 25 or more than 40"):
        smote = SMOTE()  # Oversampling method
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        return X_train_smote, y_train_smote
    else:
        st.write("The dataset is not artificially balanced")
        return X_train, y_train

def choice_auto_ML_selection(X_train, X_test, y_train, y_test):
# Define the parameter grid to search over

    models = [
            "LogisticRegression",
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "SVC",
            "KNeighborsClassifier"
        ]
    option = st.radio('Select the classification model to use:', models)
    if option == "LogisticRegression":
        model=LogisticRegression()
        list_iter= st.text_input('Enter the list of max iterations to test',[100, 10])
        list_iter=eval(list_iter)
        list_solver = st.multiselect("What solver do you want to use?",options=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],default=['lbfgs'])
        param_grid = {'LogisticRegression': {'max_iter': list_iter, 'solver': list_solver}}
    elif option == "DecisionTreeClassifier":
        model=DecisionTreeClassifier()
        list_depth = st.text_input('Enter the list of max depth to test',[None, 10])
        list_depth=eval(list_depth)
        list_min_samples_split =st.text_input('Enter the list of min sample split to test',[2, 10])
        list_min_samples_split=eval(list_min_samples_split)
        list_min_samples_leaf = st.text_input('Enter the list of min sample leaf to test',[1,4])
        list_min_samples_leaf = eval(list_min_samples_leaf)
        param_grid = {'DecisionTreeClassifier': {'max_depth': list_depth, 'min_samples_split': list_min_samples_split,'min_samples_leaf': list_min_samples_leaf}}
    elif option =="RandomForestClassifier":
        model =RandomForestClassifier()
        list_estimators = st.text_input("Enter a list of n estimator to test",[100, 5000])
        list_estimators=eval(list_estimators)
        list_depth = st.text_input('Enter the list of max depth to test for the forest', [None, 20])
        list_depth = eval(list_depth)
        param_grid = {'RandomForestClassifier': {'n_estimators': list_estimators, 'max_depth': list_depth}}
    elif option =="SVC":
        model =SVC()
        list_SVC = st.text_input('Enter the list of max depth to test for the SVC', [0.1, 1])
        list_SVC = eval(list_SVC)
        list_solver = st.multiselect("What kernel do you want to use?",options=['linear', 'rbf'], default=['linear'])
        param_grid = {'SVC': {'C': list_SVC, 'kernel': list_solver}}
    elif option =="KNeighborsClassifier":
        model=KNeighborsClassifier()
        list_n_neighbors = st.text_input('Enter the list of n neighbors to test ', [3, 5])
        list_n_neighbors = eval(list_n_neighbors)
        list_weights = st.multiselect("What weights do you want to use?",options=['uniform', 'distance'], default=['uniform', 'distance'])
        param_grid = {'KNeighborsClassifier':{
        'n_neighbors': list_n_neighbors, 'weights': list_weights}}

    if st.checkbox("Launch de analysis"):

    # models to test: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), Naive Bayes, K-Nearest Neighbors (KNN)
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



    """for model in models:
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
        #pickle.dump(model, open(filename, 'wb'))"""