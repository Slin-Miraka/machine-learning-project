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



st.title('**机器学习交互app (demo)**')

with st.sidebar.header('Welcome！ o(*￣▽￣*)ブ'):
    dataset_name = st.sidebar.selectbox(
        '选择一个你喜欢的数据集',
        ('鸢尾花数据集', '乳腺癌数据集', '葡萄酒数据集')
    )

    classifier_name = st.sidebar.selectbox('选择分类器',('决策树','LazyClassifier','KNN', 'SVM', 'Random Forest'))
    

with st.sidebar.header('基础参数'):
    parameter_random_state = st.sidebar.slider('随机种子个数(目的：结果复现)', 0, 1000, 42, 1)
    split_size = st.sidebar.slider('数据拆分比例 (设置训练集的百分比)', 10, 90, 80, 5)
    
def add_parameter_ui():  
    params = dict()
    with st.sidebar.header('模型参数'):
        params['criterion'] = st.sidebar.selectbox('评价每次决策树决策的分枝质量的指标，即衡量不纯度的指标输入"gini"使用基尼系数，或输入"entropy" 使用信息增益(Information Gain)',('gini', 'entropy'))
        params['splitter'] = st.sidebar.selectbox('确定每个节点的分枝策略输入"best"使用最佳分枝，或输入"random"使用最佳随机分枝',('best', 'random'))
        params['max_features'] = st.sidebar.selectbox('在做最佳分枝的时候，考虑的特征个数', options=['auto', 'sqrt', 'log2'])  
    with st.sidebar.header('模型修正参数'):
        params['max_depth'] = st.sidebar.slider('树的最大深度',1, 10, 4, 1)
        params['min_samples_split'] = st.sidebar.slider('一个中间节点要分枝所需要的最小样本量', 1, 10, 4, 1)
        params['min_samples_leaf'] = st.sidebar.slider('一个叶节点要存在所需要的最小样本量', 1, 10, 4, 1)
        return params
 
@st.cache(suppress_st_warning=True)
def get_dataset(name):
    data = None
    if name == '鸢尾花数据集':
        data = datasets.load_iris()
    elif name == '葡萄酒数据集':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

def get_target(name):
    target = None
    if name == '鸢尾花数据集':
        target = ['山鸢尾','变色鸢尾','维吉尼亚鸢尾']
    elif name == '葡萄酒数据集':
        target = ['葡萄酒_1','葡萄酒_2','葡萄酒_3']
    else:
        target = ['恶性肿瘤','良性肿瘤']
    return target

def get_feature(name):
    feature = None
    if name == '鸢尾花数据集':
        feature = ['花萼长度(cm)','花萼宽度(cm)','花瓣长度(cm)','花瓣宽度(cm)']
    elif name == '葡萄酒数据集':
        feature = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
    else:
        feature = ['平均半径','平均纹理','平均周长','平均面积','平均光滑度','平均紧凑度','平均凹陷','平均凹点','平均对称性','平均分形维度'
                   ,'半径误差','纹理误差','周长误差','面积误差','光滑度误差','紧凑度误差','凹陷误差','凹点误差','对称性误差','分形尺寸误差'
                   ,'最差半径','最差纹理','最差周长','最差面积','最差平滑度','最差紧凑度','最差凹陷','最差凹点','最差对称性','最差分形尺寸']
    return feature



T = get_target(dataset_name)
F = get_feature(dataset_name)
F_1 = ["标签(target)"]

X, y = get_dataset(dataset_name)

df = pd.concat([pd.DataFrame(X,columns = F),pd.DataFrame(y,columns = F_1)],axis = 1)


st.write('**Edit by:** Miraka Lin')
#st.write('**Email:** slin3137@uni.sydney.edu.au')
st.subheader('**1.数据展示**')
st.write('展示数据集的**前5行**')
st.write(df.head(5))
st.write('**数据结构:**')
st.info(df.shape)
st.write('**标签类别:**')
st.info(T)
st.write('**描述性统计**')
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
    
    st.subheader('**2.数据可视化**')
    st.write('**决策树 ** --当前叶子的个数:',clf.get_n_leaves())
    st.graphviz_chart(dot_data)
    
    ##Feature importance plot
    feature_column = ["特征","重要性打分"]
    feature_data = pd.DataFrame([*zip(F,clf.feature_importances_)],columns =feature_column)
    feature_data = feature_data.sort_values(by = ["重要性打分"],ascending=False)
    feature_data = feature_data.loc[feature_data['重要性打分']>0]
    st.write('**用于分类的重要特征**'
             #,feature_data.loc[feature_data['Important_Socre']>0]
             )
    #st.info(pd.DataFrame([*zip(F,clf.feature_importances_)]))
    plt.figure(figsize=(6,3))
    #sns.set_style(style="whitegrid")
    #plt.rcParams['font.sans-serif']=['SimHei','Arial']
    #plt.rcParams['axes.unicode_minus'] = False
    #sns.set(font=['SimHei','Arial'], font_scale=0.8)
    sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']}) 

    ax = sns.barplot(y="特征", x="重要性打分", data=feature_data)
    fig = ax.get_figure()
    st.pyplot(fig)
    st.write('**模型预测准确率:**', score)
        
Decision_Tree_Visualisation()

def Decision_Boundary_plot():
    ##Decision boundary plot
    
    n_classes = len(np.unique(y))
    color_t = ["red","blue","orange","green"]
    colors = random.sample(color_t, len(np.unique(y)))
    plot_colors = colors
    plot_step = 0.02
    st.write('**决策边界图**-- 使用一对分类重要特征')
    feature_column = ["Features","Important_Socre"]
    feature_data = pd.DataFrame([*zip(F,clf.feature_importances_)],columns =feature_column)
    feature_data = feature_data.sort_values(by = ["Important_Socre"],ascending=False)
    feature_data = feature_data.loc[feature_data['Important_Socre']>0]
    feature_B = list(feature_data.iloc[:,0])
    x_B = st.selectbox('选择X轴',feature_B)
    y_B = st.selectbox('选择Y轴',feature_B)
    
    if x_B == y_B:
        st.write("请选择不同的**X**和**Y**  **!**")
    else:
        fig = plt.figure(figsize=(6,3))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
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
    
        plt.suptitle("由一对重要特征构建的决策数决策边界图")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        st.pyplot(fig)

Decision_Boundary_plot()