import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("streanlit example")
st.write("""
        # Explore Classifiers
        Which one is the best?
        """)
ds_name = st.sidebar.selectbox("select dataset", ("Iris", "BrCan", "Wine"))
cls_name = st.sidebar.selectbox("select clst", ("KNN", "SVM", "Random Forest"))

def load_ds(dsname):
    if ds_name == "Iris":
        data = datasets.load_iris()
    elif ds_name == "BrCan":
        dara = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target

    return x,y

X,y = load_ds(ds_name)
st.write("X shape: ", X.shape)
st.write("No of classes = ", len(np.unique(y)))

pars = dict()

def add_param_ui(cls):
    #pars = dict()

    if cls == "KNN":
        K = st.sidebar.slider("K", 1,15)
        pars["K"] = K
    elif cls == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        pars["C"] = C 
    else:
        max_depth = st.sidebar.slider("max_depth", 2,15)
        n_est = st.sidebar.slider("n_est", 1,100)
        pars["max_depth"] = max_depth
        pars["n_est"] = n_est

        return pars
    
params = add_param_ui(cls_name)

def get_cls(cls_name, params):
    if cls_name == "KNN":
        cls = KNeighborsClassifier(n_neighbors = params["K"])
    elif cls_name == "SVM":
        cls = SVC(C= params["C"])
    else:
        cls = RandomForestClassifier(n_estimators=params["n_est"], max_depth=params["max_depth"],random_state=123)

    return cls

clf = get_cls(cls_name, pars)

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"clf = {cls_name}")
st.write(f"Acc = {acc}")

pca = PCA(2)
X_prj = pca.fit_transform(X)  

x1 = X_prj[ : ,0]
x2 = X_prj[ : ,1] 

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel(" PCA 1")
plt.ylabel(" PCA 2")
plt.colorbar()

st.pyplot(fig)

