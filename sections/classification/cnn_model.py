import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import streamlit as st
from tensorflow.keras.utils import to_categorical
from sections.classification.ML_model import standardization_features
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def choice_standardization_features2(df, list_columns):
    df2 = pd.DataFrame()

    if st.checkbox("Standardize some columns for CNN?"):
        if st.checkbox("Standardize all columns for CNN?"):
            for col in list_columns:
               df2[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            list_col_to_std = st.multiselect("Define the features to standardize for CNN", options=list_columns,
                                              default=list_columns)
            for col in list_col_to_std:
                df2[col] = (df[col] - df[col].mean()) / df[col].std()
    else:
        df2 = df.copy()
    df2["target"]=df["target"]
    return df2
def cnn_modeling(df,list_columns):
    df_norm = choice_standardization_features2(df, list_columns)
    #Select parameter for the CNN, number of layers, number of neurons, type of activation function
    nb_of_layers = st.slider('How many layers of neurons?',
        4, 8, 4,1
    )
    nb_of_neurons = st.slider('How many neurons per layer?',
                      16, 128, 64, 8
                      )
    type_data = st.radio(
        "Choice you activation function",
        ["relu", "sigmoid", "tanh"]
    )


# Split the data into X (features) and y (target)
    X = df_norm.drop(columns=['target'])
    y = df_norm['target']

    # First, split the data into training and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Now, split the temp data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Now you have the splits: X_train, X_val, X_test, y_train, y_val, y_test

    model = Sequential()

    # Input layer (number of neurons = number of features in your data)
    model.add(Dense(nb_of_neurons, activation=type_data, input_shape=(X_train.shape[1],)))
    nb_of_layers_to_implement = nb_of_layers -2
    #1st hidden layer
    model.add(Dense(int(nb_of_neurons / 2), activation=type_data))

    for i in range(nb_of_layers_to_implement):
    # Hidden layers
        model.add(Dropout(0.5))  # Dropout to reduce overfitting
        model.add(Dense(int(nb_of_neurons/(2**(i+1))), activation=type_data))


    # Output layer (3 neurons for 3 classes, with softmax activation)
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    y_train_encoded = to_categorical(y_train, num_classes=3)
    y_val_encoded = to_categorical(y_val, num_classes=3)

    y_test_encoded = to_categorical(y_test, num_classes=3)
    history = model.fit(X_train, y_train_encoded,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val, y_val_encoded))

    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
    st.write(f"Test Accuracy: {test_accuracy:.4f}")
    #Generate a classification report

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)


    report_dict = classification_report(y_test, y_pred_classes, output_dict=True)  # Set output_dict=True to return a dictionary
    df_classification_report = pd.DataFrame(report_dict)
    st.write(f"The model has the following performance:")
    st.write(df_classification_report)

    with st.expander("See confusion matrix"):
        #Generate a confusion matrix
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_test, y_pred_classes)

        #plot the confusion matrix
        from sklearn.metrics import confusion_matrix

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix")

        # Show the plot in Streamlit
        st.pyplot(fig)  # Now we pass the created figure object to Streamlit