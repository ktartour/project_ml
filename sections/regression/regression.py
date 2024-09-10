import streamlit as st
import pandas as pd
import io
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def regression_page():
    # Function to load and store the DataFrame in st.session_state
    def load_data(file: io.BytesIO) -> pd.DataFrame:
        if 'data' not in st.session_state:
            st.session_state['data'] = pd.read_csv(file)
        return st.session_state['data']

    # Function to update the columns in st.session_state
    def update_columns(df: pd.DataFrame, new_columns: List[str]) -> None:
        df.columns = new_columns
        st.session_state['renamed'] = True
        st.session_state['data'] = df

    # Configuration of the interface with tabs
    tabs = st.tabs(["Upload File",
                    "DataFrame Analysis",
                    "Standardization Check",
                    "Visualization and Correlations",
                    "Regression Models",
                    "Looking for best parameters"])

    # First tab: File selection
    with tabs[0]:
        st.write('# Regression Models')
        st.write('#### Select a file to upload.')

        uploaded_file = st.file_uploader(' ', type=['csv', 'txt'], accept_multiple_files=False, key="Laurent")

        if uploaded_file is not None:
            # Load the data
            df: pd.DataFrame = load_data(uploaded_file)
            st.success('File uploaded successfully.')

        # URL of the image
        image_url: str = 'https://miro.medium.com/v2/resize:fit:3960/format:webp/1*Cgqkt7lTWELcNV4v3KYsRw.png'

        # Display the image across the full width of the page
        st.image(image_url, use_column_width=True)

    # Second tab: DataFrame Analysis
    with tabs[1]:
        if uploaded_file is not None:
            # Display DataFrame dimensions
            st.subheader("DataFrame Format")
            st.write(df.shape)

            # Display column names in a line
            st.subheader("Column Names")
            st.write(', '.join(df.columns))

            # Renaming Columns for diabete.csv file
            if uploaded_file.name == "diabete.csv":
                st.subheader("Renaming Columns")

                # Define new column names
                new_names: List[str] = [
                    'Unnamed: 0',
                    'Age',
                    'Sex',
                    'Body Mass Index',
                    'Mean Arterial Pressure',
                    'Serum Cholesterol',
                    'Low-Density Lipoproteins',
                    'High-Density Lipoproteins',
                    'Total Cholesterol',
                    'Log of Serum Triglycerides',
                    'Blood Sugar',
                    'target'
                ]

                # Matching before and after transformation
                if len(df.columns) == len(new_names):
                    st.write("Column name mapping before and after transformation:")
                    col_mapping: pd.DataFrame = pd.DataFrame({
                        'Before': df.columns,
                        'After': new_names
                    })
                    st.write(col_mapping)

                    # Add "Validate" and "Do Not Validate" buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Validate"):
                            update_columns(df, new_names)
                            st.session_state['button_clicked'] = True

                    with col2:
                        if st.button("Do Not Validate"):
                            st.session_state['button_clicked'] = True

            # If any button was clicked, display the following sections
            if st.session_state.get('button_clicked', False):

                # Display column types
                st.subheader("Column Types")
                st.write(df.dtypes)

                # Display DataFrame statistics
                st.subheader("DataFrame Statistics")
                st.write(df.describe())

                # Display first rows of the DataFrame
                st.subheader("DataFrame Head")
                st.write(df.head())

                # Display DataFrame information
                st.subheader("DataFrame Info")
                buffer: io.StringIO = io.StringIO()
                df.info(buf=buffer)
                s: str = buffer.getvalue()
                st.text(s)

                # Display columns containing NaN values
                st.subheader("Columns with NaNs")
                nan_col: pd.Series = df.isna().sum()
                if not nan_col[nan_col > 0].empty:
                    st.write(nan_col[nan_col > 0])  # Filter to show only columns with NaNs
                else:
                    st.write("There are no NaN values")

                # Display all columns with the possibility to delete using the page width
                st.subheader("All Columns with Delete Option")
                cols: List[str] = df.columns.tolist()
                cols_per_row: int = 4  # Number of columns per row

                # Calculate the number of rows needed
                num_rows: int = len(cols) // cols_per_row + (len(cols) % cols_per_row > 0)

                # Use st.columns to create a grid of buttons
                for row in range(num_rows):
                    cols_in_row: List[str] = cols[row * cols_per_row:(row + 1) * cols_per_row]
                    cols_row = st.columns(len(cols_in_row))
                    for i, col in enumerate(cols_in_row):
                        with cols_row[i]:
                            if st.button(f'Delete Column: {col}', key=f'all_{col}'):
                                df.drop(columns=[col], inplace=True)

                # Display updated DataFrame (only first few rows) across the entire page width
                st.subheader("Updated DataFrame")
                st.write(df.head())

                # Display rows with NaNs
                st.subheader("Rows with NaNs")
                rows_with_nans: pd.DataFrame = df[df.isna().any(axis=1)]  # Filter to show rows with any NaN values
                if not rows_with_nans.empty:
                    st.write(rows_with_nans)  # Display the rows with NaNs
                else:
                    st.write("There are no NaN values")

                # Add section for handling rows with NaNs
                st.subheader("Options for Handling NaN Rows")
                col1, col2, col3, col4 = st.columns(4)

                # Option 1: Replace NaNs with mean
                with col1:
                    if st.button('Replace with Mean'):
                        df.fillna(df.mean(), inplace=True)
                        st.success("NaNs have been replaced with the mean.")

                # Option 2: Replace NaNs with median
                with col2:
                    if st.button('Replace with Median'):
                        df.fillna(df.median(), inplace=True)
                        st.success("NaNs have been replaced with the median.")

                # Option 3: Replace NaNs with 0
                with col3:
                    if st.button('Replace with 0'):
                        df.fillna(0, inplace=True)
                        st.success("NaNs have been replaced with 0.")

                # Option 4: Drop rows containing NaNs
                with col4:
                    if st.button('Drop Rows with NaNs'):
                        df.dropna(inplace=True)
                        st.success("Rows with NaNs have been dropped.")

    with tabs[2]:
        if uploaded_file is not None:
            st.subheader("Column Standardization Check")

            # Initialization of lists to store results
            standardization_results: list[dict[str, str | float]] = []

            for column in df.drop('target', axis=1).columns:
                mean: float = df[column].mean()
                std: float = df[column].std()
                min_val: float = df[column].min()
                max_val: float = df[column].max()

                # Checking for standardization
                standardization_z: bool = (round(mean, 3) == 0 and std == 1)
                standardization_mm: bool = (round(mean, 3) == 0 and min_val >= 0 and max_val <= 1)
                standardization_robust: bool = not standardization_z and not standardization_mm

                standardization_type: str = (
                    'Z-score Standardization' if standardization_z
                    else 'Min-Max Standardization' if standardization_mm
                    else 'Robust Standardization' if standardization_robust
                    else 'Not Standardized'
                )

                standardization_results.append({
                    'Column': column,
                    'Mean': mean,
                    'Std Dev': std,
                    'Min': min_val,
                    'Max': max_val,
                    'Z-score Standardization': 'Yes' if standardization_z else 'No',
                    'Min-Max Standardization': 'Yes' if standardization_mm else 'No',
                    'Robust Standardization': 'Yes' if standardization_robust else 'No'
                })

            # Creating a DataFrame for standardization results
            df_standardization: pd.DataFrame = pd.DataFrame(standardization_results)
            st.write(df_standardization)

            # Adding the section for the Shapiro-Wilk test
            st.subheader("Shapiro-Wilk Test for Normality")

            shapiro_results: list[dict[str, float | str]] = []
            for column in df.columns:
                stat: float
                p: float
                stat, p = shapiro(df[column].dropna())  # dropna() pour éviter les problèmes avec les NaN
                shapiro_results.append({
                    'Column': column,
                    'Test Statistic': stat,
                    'P-value': p,
                    'Normality': 'Yes' if p > 0.05 else 'No'
                })

            # Creating a DataFrame for Shapiro-Wilk test results
            df_shapiro: pd.DataFrame = pd.DataFrame(shapiro_results)
            st.write(df_shapiro)

    with tabs[3]:
        if uploaded_file is not None:
            st.subheader("Column Distribution Plot")
            features: list[str] = st.multiselect(
                'Select columns to display distribution:',
                df.columns.tolist()
            )

            # Displaying the distribution for selected columns
            if features:
                for column in features:
                    st.write(f"### Distribution of {column}")
                    fig, ax = plt.subplots()
                    sns.histplot(df[column], kde=True, ax=ax)
                    plt.title(f'Distribution of {column}')
                    st.pyplot(fig)

            st.subheader("Correlation Search")
            fig, ax = plt.subplots(figsize=(9, 5))
            mask: np.ndarray = np.triu(df.corr())
            cmap = sns.diverging_palette(15, 160, n=11, s=100)

            sns.heatmap(
                df.corr(),
                mask=mask,
                annot=True,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                ax=ax
            )
            st.pyplot(fig)

            # Selecting the correlation threshold
            threshold: float = st.slider('Select correlation threshold:', 0.0, 1.0, 0.4)
            st.subheader("Correlations Above the Threshold")
            correlations: pd.Series = df.corr()['target'].drop('target')
            features_above_threshold: dict[str, float] = {feature: correlation for feature, correlation in
                                                          correlations.items() if abs(correlation) > threshold}

            if features_above_threshold:
                correlation_df: pd.DataFrame = pd.DataFrame(list(features_above_threshold.items()),
                                                            columns=['Feature', 'Correlation']).set_index('Feature')
                st.write(correlation_df)
            else:
                st.write("No correlations above the selected threshold.")

            # Pairplot visualization
            st.subheader("Pairplot Visualization")
            pairplot_features: list[str] = st.multiselect(
                'Select columns for the pairplot:',
                df.columns.tolist()
            )

            if pairplot_features:
                st.write("### Pairplot of selected columns")
                fig = sns.pairplot(df[pairplot_features])
                st.pyplot(fig)

            # Searching for collinearity (Variance Inflation Factor - VIF)
            st.subheader("Collinearity Search")
            X: pd.DataFrame = df.drop('target', axis=1)
            vif: pd.DataFrame = pd.DataFrame()
            vif["Feature"] = X.columns
            vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

            st.write("### VIF (Variance Inflaction Factor) of Features")
            st.dataframe(vif)

            # Filtering collinearities
            st.subheader("Filter Collinearities")
            collinearity_option: str = st.selectbox(
                'Select the type of collinearity to display:',
                ['Non-Collinear (VIF = 1)', 'Moderately Collinear (1 < VIF ≤ 5)', 'Highly Collinear (VIF > 5)']
            )

            filtered_vif: pd.DataFrame
            if collinearity_option == 'Non-Collinear (VIF = 1)':
                filtered_vif = vif[vif['VIF'] == 1]
            elif collinearity_option == 'Moderately Collinear (1 < VIF ≤ 5)':
                filtered_vif = vif[(vif['VIF'] > 1) & (vif['VIF'] <= 5)]
            elif collinearity_option == 'Highly Collinear (VIF > 5)':
                filtered_vif = vif[vif['VIF'] > 5]

            # Checking if the filtered DataFrame is empty
            if filtered_vif.empty:
                st.write("No features match this VIF.")
            else:
                st.write(filtered_vif)

            # Feature selector for processing
            st.subheader("Select Features for Processing")
            features: list[str] = [col for col in df.columns if col != 'target']
            selected_features: list[str] = [feature for feature in features if st.checkbox(feature, value=True)]

            # Adding the 'Target' column to the list of selected features
            selected_features.append('target')

            # Creating a new DataFrame with the selected features
            df_selected: pd.DataFrame = df[selected_features]

            # Displaying the updated DataFrame (only the first few rows)
            st.subheader("Updated DataFrame")
            st.write(df_selected.head())

    with tabs[4]:
        if uploaded_file is not None:
            st.subheader("Regressions")

            # Splitting the data
            test_size: float = st.slider('Select test set size:', 0.1, 0.4, 0.3, 0.1)
            X: pd.DataFrame = df_selected.drop('target', axis=1)
            y: pd.Series = df_selected['target']
            X_train: pd.DataFrame
            X_test: pd.DataFrame
            y_train: pd.Series
            y_test: pd.Series
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Selecting the number of folds
            cv_folds: int = st.slider('Select number of folds for cross-validation:', 2, 5, 3)

            # List of models
            models: list[tuple[str, object]] = [
                ('Linear Regression', LinearRegression()),
                ('Ridge', Ridge()),
                ('Lasso', Lasso()),
                ('Bayesian Ridge Regression', BayesianRidge()),
                ('ElasticNet', ElasticNet()),
                ('Decision Tree', DecisionTreeRegressor()),
                ('SVR', SVR())
            ]

            # Evaluating the models
            results: list[dict[str, float]] = []
            for name, model in models:
                model.fit(X_train, y_train)
                y_pred: np.ndarray = model.predict(X_test)

                # Calculate metrics
                rmse: float = np.sqrt(mean_squared_error(y_test, y_pred))
                r2: float = r2_score(y_test, y_pred)
                mae: float = mean_absolute_error(y_test, y_pred)

                # Cross-validation
                rmse_cv: np.ndarray = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
                r2_cv: np.ndarray = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                mae_cv: np.ndarray = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')

                # Storing the results
                results.append({
                    'Model': name,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAE': mae,
                    'Cross-Validated RMSE': np.mean(rmse_cv),
                    'Cross-Validated R2': np.mean(r2_cv),
                    'Cross-Validated MAE': np.mean(mae_cv)
                })

            # Creating a DataFrame to display the results
            df_results: pd.DataFrame = pd.DataFrame(results)
            st.write(df_results)

            # Polynomial Regression
            st.subheader("Polynomial Regression")
            degree_min = st.slider("Choose the lowest polynomial degree:", 1, 10, 1)
            degree_max = st.slider("Choose the highest polynomial degree:", 1, 10, 3)

            poly_results = []
            for degree in range(degree_min, degree_max + 1):
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression(fit_intercept=False))
                ])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                # Cross-validation
                rmse_cv = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_root_mean_squared_error')
                r2_cv = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                mae_cv = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')

                poly_results.append({
                    'Degree': degree,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAE': mae,
                    'Cross-Validated RMSE': rmse_cv.mean(),
                    'Cross-Validated R2': r2_cv.mean(),
                    'Cross-Validated MAE': mae_cv.mean()
                })

            # Display polynomial regression results as a table
            st.subheader("Polynomial Regression Results")
            poly_results_df = pd.DataFrame(poly_results)
            st.write(poly_results_df)

    with tabs[5]:

        if uploaded_file is not None:
            st.subheader("Looking for best parameters")

            # Sélection du scoring
            scoring_options = {
                'RMSE': 'neg_root_mean_squared_error',
                'R2': 'r2',
                'MAE': 'neg_mean_absolute_error'
            }
            scoring = st.selectbox('Select scoring metric for GridSearchCV:', list(scoring_options.keys()))
            scoring_metric = scoring_options[scoring]

            # Splitting the data
            test_size_gs: float = st.slider('Select test set size:', 0.1, 0.4, 0.3, 0.1, key='gs')
            X: pd.DataFrame = df_selected.drop('target', axis=1)
            y: pd.Series = df_selected['target']
            X_train: pd.DataFrame
            X_test: pd.DataFrame
            y_train: pd.Series
            y_test: pd.Series
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_gs, random_state=42)

            # Selecting the number of folds
            cv_folds_gs: int = st.slider('Select number of folds for cross-validation:', 2, 5, 3, key='gs2')

            # List of models with parameter grids
            models: list[tuple[str, object, dict]] = [
                ('Linear Regression', LinearRegression(), {}),
                ('Ridge', Ridge(), {'alpha': [0.1, 1.0, 10.0]}),
                ('Lasso', Lasso(), {'alpha': [0.1, 1.0, 10.0]}),
                ('Bayesian Ridge Regression', BayesianRidge(), {'alpha_1': [0.1, 1.0, 10.0]}),
                ('ElasticNet', ElasticNet(), {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}),
                ('Decision Tree', DecisionTreeRegressor(),
                 {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),
                ('SVR', SVR(), {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']})
            ]

            # Evaluating the models with Grid Search
            results: list[dict[str, float]] = []
            for name, model, param_grid in models:
                grid_search = GridSearchCV(model, param_grid, cv=cv_folds_gs, scoring=scoring_metric, n_jobs=-1)
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                y_pred: np.ndarray = best_model.predict(X_test)

                # Calculate metrics
                rmse: float = np.sqrt(mean_squared_error(y_test, y_pred))
                r2: float = r2_score(y_test, y_pred)
                mae: float = mean_absolute_error(y_test, y_pred)

                # Cross-validation
                rmse_cv: np.ndarray = -cross_val_score(best_model, X, y, cv=cv_folds_gs,
                                                       scoring='neg_root_mean_squared_error')
                r2_cv: np.ndarray = cross_val_score(best_model, X, y, cv=cv_folds_gs, scoring='r2')
                mae_cv: np.ndarray = -cross_val_score(best_model, X, y, cv=cv_folds_gs,
                                                      scoring='neg_mean_absolute_error')

                # Storing the results
                results.append({
                    'Model': name,
                    'Best Params': grid_search.best_params_,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAE': mae,
                    'Cross-Validated RMSE': np.mean(rmse_cv),
                    'Cross-Validated R2': np.mean(r2_cv),
                    'Cross-Validated MAE': np.mean(mae_cv)
                })

            # Creating a DataFrame to display the results
            df_results: pd.DataFrame = pd.DataFrame(results)
            st.write(df_results)

            st.subheader("Polynomial Regression")

            degree_min = st.slider("Choose the lowest polynomial degree:", 1, 10, 1, key="gs4")
            degree_max = st.slider("Choose the highest polynomial degree:", 1, 10, 3, key="gs5")

            # Selecting the number of folds
            cv_folds_gs2: int = st.slider('Select number of folds for cross-validation:', 2, 5, 3, key='gs3')

            # Define the parameter grid for GridSearchCV
            param_grid = {
                'poly__degree': list(range(degree_min, degree_max + 1))
            }

            # Create a pipeline that includes polynomial features and linear regression
            pipeline = Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LinearRegression(fit_intercept=False))
            ])

            # Setup GridSearchCV
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv_folds_gs2, scoring=scoring_metric, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get the best model from GridSearchCV
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Evaluate the best model on the test set
            y_pred = best_model.predict(X_test)

            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Cross-validation
            rmse_cv = -cross_val_score(best_model, X, y, cv=cv_folds_gs2, scoring='neg_root_mean_squared_error')
            r2_cv = cross_val_score(best_model, X, y, cv=cv_folds_gs2, scoring='r2')
            mae_cv = -cross_val_score(best_model, X, y, cv=cv_folds_gs2, scoring='neg_mean_absolute_error')

            # Prepare results
            poly_results = {
                'Best Degree': best_params['poly__degree'],
                'RMSE': rmse,
                'R2': r2,
                'MAE': mae,
                'Cross-Validated RMSE': rmse_cv.mean(),
                'Cross-Validated R2': r2_cv.mean(),
                'Cross-Validated MAE': mae_cv.mean()
            }

            # Display polynomial regression results as a table
            st.subheader("Polynomial Regression Results")
            poly_results_df = pd.DataFrame([poly_results])
            st.write(poly_results_df)

    return