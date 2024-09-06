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

def standardization_features2(df, list_columns):
    df2= pd.DataFrame()
# standardization of columns of list_to_norm
    for col in list_columns:
        df2[col] = (df[col] - df[col].mean()) / df[col].std()

    df2["target"]=df["target"]
    return df2
def cnn_modeling(df,list_columns):
    df_norm = standardization_features2(df, list_columns)
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
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

    # Hidden layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to reduce overfitting
    model.add(Dense(16, activation='relu'))

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
    st.write(classification_report(y_test, y_pred_classes))
    #Generate a confusion matrix
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)

    #plot the confusion matrix
    from sklearn.metrics import confusion_matrix

    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)
