import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import matplotlib.pyplot as plt

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous ?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")
    
    
    def load_data():
        data = pd.read_csv("E:\PYTHON\streamlit-ml\mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    
    def split(df):
        X = df.drop("type", axis=1)
        y = df.type
        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=101)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            figure = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred, labels=model.classes_), display_labels=model.classes_)
            figure.plot(ax=ax)
            st.pyplot(fig)
            
            
        
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            figure = RocCurveDisplay(fpr=fpr, tpr=tpr)
            figure.plot(ax=ax)
            st.pyplot(fig)

        if "Precision Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            figure = PrecisionRecallDisplay(precision=precision, recall=recall)
            figure.plot(ax=ax)
            st.pyplot(fig)

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization paramter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
        if st.sidebar.button("Classify", key='CLASSIFY'):
            st.subheader("Support Vector Machine (SVM)")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(X_train, y_train)
            acccuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write(f"Accuracy : {round(acccuracy, 2)}")
            st.write(f"Precision score : {round(precision_score(y_test, y_pred, labels=class_names),2)}")
            st.write(f"Recall Score : {recall_score(y_test, y_pred)}")
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization paramter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
        if st.sidebar.button("Classify", key='CLASSIFY'):
            st.subheader("Logistic Regression")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            acccuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write(f"Accuracy : {round(acccuracy, 2)}")
            st.write(f"Precision score : {round(precision_score(y_test, y_pred, labels=class_names),2)}")
            st.write(f"Recall Score : {recall_score(y_test, y_pred)}")
            plot_metrics(metrics)

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 500, step=10, key='N-estimators')
        max_depth =st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap')


        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))
        if st.sidebar.button("Classify", key='CLASSIFY'):
            st.subheader("Random Forest")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
            model.fit(X_train, y_train)
            acccuracy = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            st.write(f"Accuracy : {round(acccuracy, 2)}")
            st.write(f"Precision score : {round(precision_score(y_test, y_pred, labels=class_names),2)}")
            st.write(f"Recall Score : {recall_score(y_test, y_pred)}")
            plot_metrics(metrics)
            
            
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.dataframe(df)

if __name__ == '__main__':
    main()


