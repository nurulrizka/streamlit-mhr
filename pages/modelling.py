import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, svm
from component.nav import navbar

# ------------------navbar-----------------------
navbar()

def model():
    # ------------------Modelling-----------------------
    st.markdown(
        '<h1 align="center">Modelling</h1>',
        unsafe_allow_html=True
    )
    
    
    import warnings
    warnings.filterwarnings("ignore")


    maternal_health_risk = fetch_ucirepo(id=863)

    X = maternal_health_risk.data.features
    y = maternal_health_risk.data.targets


    # Convert features and targets to pandas DataFrame
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    # Add an 'ID' column to X_df
    # X_df.insert(0, 'ID', X_df.index)

    # Concatenate features and targets horizontally
    maternal_health_risk = pd.concat([X_df, y_df], axis=1)


    # Rename columns for better readability
    maternal_health_risk.rename(columns={
        'BS': 'BloodSugar'
    }, inplace=True)


    # Save DataFrame to CSV file
    maternal_health_risk.to_csv('maternal_health_risk.csv', index=False)

    numerical_columns = [col for col in maternal_health_risk.columns if maternal_health_risk[col].dtype != 'O']
    categorical_columns = [col for col in maternal_health_risk.columns if maternal_health_risk[col].dtype == 'O']
    discrete_columns = [col for col in maternal_health_risk.columns if len(maternal_health_risk[col].unique())<7]
    continuous_columns = [col for col in maternal_health_risk.columns if len(maternal_health_risk[col].unique())>10]

    df_main = maternal_health_risk.copy()

    for col in categorical_columns:
        le = LabelEncoder()
        df_main[col] = le.fit_transform(df_main[col])


    df_main_lof = df_main.copy()

    # Menggunakan semua kolom fitur untuk deteksi outlier
    columns_for_lof = list(X.columns)

    # Inisialisasi model LOF
    lof = LocalOutlierFactor(n_neighbors=40, p=2)

    # Mendeteksi outlier
    outlier_scores = lof.fit_predict(X)
    df_main_lof['LOF_Score'] = -lof.negative_outlier_factor_


    df_main_lof.insert(0, 'ID', df_main_lof.index)
    data_outlier = df_main_lof.loc[df_main_lof['LOF_Score'] > 2]
    df_main_lof = df_main_lof.loc[df_main_lof['LOF_Score'] <= 2]
    data_outlier = df_main_lof.loc[df_main_lof['LOF_Score'] > 2]
    
    # Preprocessing: Scaling and Splitting data
    features = maternal_health_risk.drop(columns=['RiskLevel'])
    target = maternal_health_risk['RiskLevel']
    
    # Scaling features using MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=5)

    # Save the scaler to be used for future scaling in deployment
    joblib.dump(scaler, 'scaler.pkl')
    
    # ------------------K-Nearest Neighbors Model-----------------------
    st.markdown("### K-Nearest Neighbors (KNN)")
    knn = KNeighborsClassifier(n_neighbors = 43, p = 1, weights = 'distance')
    knn_model = knn.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    joblib.dump(knn_model, 'knn.pkl')
    
    # Evaluate KNN model
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    st.write(f"Akurasi KNN: {accuracy_knn:.2%}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_knn))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred_knn))
    
    # ------------------Support Vector Machine Model-----------------------
    st.markdown("### Support Vector Machine (SVM)")
    svm_model = SVC(kernel='rbf', C=10, gamma=10)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    joblib.dump(svm_model, 'svm.pkl')
    
    # Evaluate SVM model
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    st.write(f"Akurasi SVM: {accuracy_svm:.2%}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_svm))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred_svm))
    
    
    # ------------------Random Forest Model-----------------------
    st.markdown("### Random Forest")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    joblib.dump(rf_model, 'rf.pkl')
    
    # Evaluate Random Forest model
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f"Akurasi Random Forest: {accuracy_rf:.2%}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_rf))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred_rf))
    
# Call the model function
model()
