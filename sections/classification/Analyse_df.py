import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def load_and_encode():
    df=pd.read_csv("data/vin.csv")

    #replace Vin éuilibré par 1, Vin amer par 0 et vin sucré par 2
    df["target"]=df["target"].replace("Vin éuilibré",1)
    df["target"]=df["target"].replace("Vin amer",0)
    df["target"]=df["target"].replace("Vin sucré",2)

    return df

def histogram_plot(df):
    # Number of columns in the DataFrame
    num_columns = df.shape[1]

    # Determine the grid size (for example, a square grid)
    ncols = 3  # You can adjust this
    nrows = ceil(num_columns / ncols)

    # Create subplots: grid of plots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

    # Flatten axes for easy iteration if it's 2D
    axes = axes.flatten()


    # Plot a histogram for each column

    for i, column in enumerate(df.columns):
        sns.histplot(df[column], bins=10, kde=True, color='skyblue', ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

# Remove any unused subplots if there are more subplots than columns
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

# Show plot
#plt.show()
    return st.pyplot(fig)

def pairplots(df,list_items):      #The list of fetures has to be a choice of the user, among column names
    # Pairplot des features selectionnés avec la variable cible
    pairplot_cols = list_items + ["target"]

    # sns.pairplot(df[pairplot_cols], hue="target", palette="viridis")
    pairplot_fig = sns.pairplot(df[pairplot_cols], hue="target")
    pairplot_fig.fig.suptitle('Pairplot of Selected Features', y=1.02)  # Adjust y for spacing

    return st.pyplot(pairplot_fig.fig)
def correlation_table(df,list_items):      #Idem list to ask
    list_features= list_items + ["target"]

    # Correlate the list of columns of the DataFrame list_items
    correlation_matrix = df[list_features].corr()
    # Create a mask to hide the upper triangle of the heatmap.
    mask = np.triu(correlation_matrix)
# Create the heatmap.
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.diverging_palette(15, 160, n=11, s=100)
    correlation_heatmap = sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    return st.pyplot(fig)

#choix de la liste de features à considérer,
# soit d'après le visuel, soit automatique d'après le coef de correlation avec la target
#Indiquer les colonnes corrélées

def explicative_columns(df):
    # Mise en avant des corrélations supérieures à un seuil donné
    l=0
    t=0
    while t==0:
        try:
            threshold = float(threshold)
            t=1
        except:
            threshold = st.text_input(
                "Let's select features correlating with the target, which threshold to use, between 0 and 1?",
                "0.5", key=l)
            l+=1

    correlations = df.corr()['target'].drop('target')
    # Initialize the list to store features with high correlation
    data = [["Feature", "Correlation","sign"]]


    # Iterate over the correlations and filter by the threshold
    for feature, correlation in correlations.items():
        if abs(correlation) > threshold:
            if correlation > 0:
                signe= "positive"
            elif correlation ==0:
                signe="null"
            elif correlation <0:
                signe = "negative"
            data.append([feature, correlation, signe])

    # Convert the list to a DataFrame with proper column names
    df_correlated_features = pd.DataFrame(data[1:], columns=data[0])
    df_correlated_features = df_correlated_features.sort_values("Correlation")
    st.write(f"Features with an absolute correlation superior to {threshold} with the target are:")
    st.write(df_correlated_features)
    return df_correlated_features
#Recherche des colinéarités


def colinearities(df, liste_columns):
    # Créer une matrice de features (sans la target)
    columns = liste_columns +["target"]
    df=df[columns]
    X = df.drop('target', axis=1)

    # Calculer le VIF (Variance inflation factor) pour chaque feature
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif=vif.sort_values(by="VIF",ascending=False)
    # Afficher les résultats
    st.write(
        "Calculation of the Variance inflation factor, it is a measure for multicollinearity of the design matrix ")
    st.write(vif)
    return vif