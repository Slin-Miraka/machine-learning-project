# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:58:23 2021

@author: ShingFoon Lin

"""

import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random



st.title('**Machine learning app (demo)**')

with st.sidebar.header('Welcome！ o(*￣▽￣*)ブ'):
    dataset_name = st.sidebar.selectbox(
        'Select Dataset',
        ('Iris', 'Breast Cancer', 'Wine')
    )

    classifier_name = st.sidebar.selectbox('Select classifier',('Decision Tree','LazyClassifier','KNN', 'SVM', 'Random Forest'))
    

with st.sidebar.header('Basic Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (train_test_split and the model)', 0, 1000, 42, 1)
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    
def add_parameter_ui():  
    params = dict()
    with st.sidebar.header('Model Parameters'):
        params['criterion'] = st.sidebar.selectbox('Criteria measuring the quality of a split (criterion)',('gini', 'entropy'))
        params['splitter'] = st.sidebar.selectbox('Split strategy at each node (splitter)',('best', 'random'))
        params['max_features'] = st.sidebar.selectbox('Max features (max_features)', options=['auto', 'sqrt', 'log2'])  
    with st.sidebar.header('Model Refining Parameters'):
        params['max_depth'] = st.sidebar.slider('The maximum depth of the tree (max_depth)',1, 10, 2, 1)
        params['min_samples_split'] = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
        params['min_samples_leaf'] = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
        return params
 
@st.cache(suppress_st_warning=True)
def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

def get_target(name):
    target = None
    if name == 'Iris':
        target = datasets.load_iris().target_names
    elif name == 'Wine':
        target = datasets.load_wine().target_names
    else:
        target = datasets.load_breast_cancer().target_names
    return target

def get_feature(name):
    feature = None
    if name == 'Iris':
        feature = datasets.load_iris().feature_names
    elif name == 'Wine':
        feature = datasets.load_wine().feature_names
    else:
        feature = datasets.load_breast_cancer().feature_names
    return feature



T = get_target(dataset_name)
F = get_feature(dataset_name)
F_1 = ["target"]

X, y = get_dataset(dataset_name)

df = pd.concat([pd.DataFrame(X,columns = F),pd.DataFrame(y,columns = F_1)],axis = 1)


st.write('**Edit by:** Miraka Lin')
#st.write('**Email:** slin3137@uni.sydney.edu.au')
st.subheader('**1.The dataset**')
st.write('Display **the first 5 row** of the dataset')
st.write(df.head(5))
st.write('**Shape of dataset**')
st.info(df.shape)
st.write('**Types of classes:**')
st.info(T)
st.write('**Statistical Description**')
st.write(df.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size,random_state = parameter_random_state)

#build model
params = add_parameter_ui()  

clf = DecisionTreeClassifier(criterion= params['criterion']
                             ,random_state= parameter_random_state
                             ,splitter= params['splitter']
                             ,max_depth = params['max_depth']
                             ,min_samples_split = params['min_samples_split']
                             ,min_samples_leaf = params['min_samples_leaf']
                             )
clf = clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

    #C = st.sidebar.slider('C', 0.01, 10.0)
    #params['C'] = C
    

def Decision_Tree_Visualisation():
    #decision tree plot
    dot_data = tree.export_graphviz(clf
                                    ,feature_names= F
                                    ,class_names= T
                                    ,filled = True
                                    ,rounded= True
                                    ,out_file= None)
    
    st.subheader('**2.Visualisation**')
    st.write('**Decision Tree ** --Current number of leaves:',clf.get_n_leaves())
    st.graphviz_chart(dot_data)
    
    ##Feature importance plot
    feature_column = ["Features","Important_Socre"]
    feature_data = pd.DataFrame([*zip(F,clf.feature_importances_)],columns =feature_column)
    feature_data = feature_data.sort_values(by = ["Important_Socre"],ascending=False)
    feature_data = feature_data.loc[feature_data['Important_Socre']>0]
    st.write('**Feature Importance**'
             #,feature_data.loc[feature_data['Important_Socre']>0]
             )
    #st.info(pd.DataFrame([*zip(F,clf.feature_importances_)]))
    plt.figure(figsize=(6,3))
    sns.set_style(style="whitegrid")
    ax = sns.barplot(y="Features", x="Important_Socre", data=feature_data)
    fig = ax.get_figure()
    st.pyplot(fig)
    st.write('**Accuracy:**', score)
        
Decision_Tree_Visualisation()

def Decision_Boundary_plot():
    ##Decision boundary plot
    
    n_classes = len(np.unique(y))
    color_t = ["red","blue","orange","green"]
    colors = random.sample(color_t, len(np.unique(y)))
    plot_colors = colors
    plot_step = 0.02
    st.write('**Decision boundary plot**-- using a pair of important variables')
    feature_column = ["Features","Important_Socre"]
    feature_data = pd.DataFrame([*zip(F,clf.feature_importances_)],columns =feature_column)
    feature_data = feature_data.sort_values(by = ["Important_Socre"],ascending=False)
    feature_data = feature_data.loc[feature_data['Important_Socre']>0]
    feature_B = list(feature_data.iloc[:,0])
    x_B = st.selectbox('Select X',feature_B)
    y_B = st.selectbox('Select Y',feature_B)
    
    if x_B == y_B:
        st.write("**X** and **Y** are expected to be the different **!**")
    else:
        fig = plt.figure(figsize=(6,3))
        Xj = pd.DataFrame(X,columns = F)
        #switch to array
        Xi = Xj.loc[:,(x_B,y_B)].values
        yi = y
        
        #st.write(Xi)
        # Train
        cllf = clf.fit(Xi, yi)
        # Plot the decision boundary
        #plt.subplot(2, 3, pairidx + 1)
        
        x_min, x_max = Xi[:, 0].min() - 1, Xi[:, 0].max() + 1
        y_min, y_max = Xi[:, 1].min() - 1, Xi[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
        Z = cllf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx
                    ,yy
                    ,Z
                    ,cmap=plt.cm.PiYG
                    ,alpha = 0.8)
    
        plt.xlabel(x_B)
        plt.ylabel(y_B)
    
        # Plot the training points
        
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(yi == i)
            plt.scatter(Xi[idx, 0]
                        ,Xi[idx, 1]
                        ,c=color
                        ,label= get_target(dataset_name)[i]
                        ,cmap=plt.cm.PiYG
                        ,edgecolor='black'
                        #,alpha = 0.3
                        ,s=15)
    
        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        st.pyplot(fig)

Decision_Boundary_plot()