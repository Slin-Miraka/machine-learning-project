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
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve



st.title('**机器学习交互app (demo)**')

with st.sidebar.header('Welcome！ o(*￣▽￣*)ブ'):
    dataset_name = st.sidebar.selectbox(
        '选择一个你喜欢的数据集',
        ('鸢尾花数据集', '乳腺癌数据集', '葡萄酒数据集')
    )

    classifier_name = st.sidebar.selectbox('选择分类器',('决策树','LazyClassifier','KNN', 'SVM', 'Random Forest'))
    

with st.sidebar.header('基础参数'):
    parameter_random_state = st.sidebar.slider('随机种子个数(目的：结果复现)', 0, 1000, 42, 1)
    split_size = st.sidebar.slider('测试数据比例 (%)', 0.1, 0.5, 0.2, 0.05)
    
def add_parameter_ui():  
    params = dict()
    with st.sidebar.header('模型参数'):
        params['criterion'] = st.sidebar.selectbox('评价每次决策树决策的分枝质量的指标，即衡量不纯度的指标输入"gini"使用基尼系数，或输入"entropy" 使用信息增益(Information Gain)',('gini', 'entropy'))
        params['splitter'] = st.sidebar.selectbox('确定每个节点的分枝策略输入"best"使用最佳分枝，或输入"random"使用最佳随机分枝',('best', 'random'))
        params['max_features'] = st.sidebar.selectbox('在做最佳分枝的时候，考虑的特征个数', options=['auto', 'sqrt', 'log2'])  
    with st.sidebar.header('模型修正参数'):
        params['max_depth'] = st.sidebar.slider('树的最大深度(max_depth)',1, 10, 4, 1)
        params['min_samples_split'] = st.sidebar.slider('一个中间节点要分枝所需要的最小样本量(min_sample_split)', 1, 10, 4, 1)
        params['min_samples_leaf'] = st.sidebar.slider('一个叶节点要存在所需要的最小样本量(min_sample_leaf)', 1, 10, 4, 1)
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

def get_target_cn(name):
    target = None
    if name == '鸢尾花数据集':
        target = ['山鸢尾','变色鸢尾','维吉尼亚鸢尾']
    elif name == '葡萄酒数据集':
        target = ['葡萄酒_1','葡萄酒_2','葡萄酒_3']
    else:
        target = ['恶性肿瘤','良性肿瘤']
    return target

def get_feature_cn(name):
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


def get_target_en(name):
    target = None
    if name == '鸢尾花数据集':
        target = datasets.load_iris().target_names
    elif name == '葡萄酒数据集':
        target = datasets.load_wine().target_names
    else:
        target = datasets.load_breast_cancer().target_names
    return target

def get_feature_en(name):
    feature = None
    if name == '鸢尾花数据集':
        feature = datasets.load_iris().feature_names
    elif name == '葡萄酒数据集':
        feature = datasets.load_wine().feature_names
    else:
        feature = datasets.load_breast_cancer().feature_names
    return feature


T_cn = get_target_cn(dataset_name)
F_cn = get_feature_cn(dataset_name)
T_en = get_target_en(dataset_name)
F_en = get_feature_en(dataset_name)

F_1 = ["标签(target)"]

X, y = get_dataset(dataset_name)

df = pd.concat([pd.DataFrame(X,columns = F_cn),pd.DataFrame(y,columns = F_1)],axis = 1)


st.write('**Edit by:** Miraka Lin')
#st.write('**Email:** slin3137@uni.sydney.edu.au')
st.subheader('**1.数据介绍**')


def data_description(name):
    if name == '鸢尾花数据集':
        st.write('''鸢尾花数据集最初由Edgar Anderson 测量得到，而后在著名的统计学家和生物学家R.A Fisher于1936年发表的文
                 章「The use of multiple measurements in taxonomic problems」中被使用，用其作为线性判别分析（Linear
                 Discriminant Analysis）的一个例子，证明分类的统计方法，从此而被众人所知，尤其是在机器学习这个领域。
                 鸢尾花数据集共收集了三类鸢尾花，即Setosa鸢尾花、Versicolour鸢尾花和Virginica鸢尾花，
                 每一类鸢尾花收集了50条样本记录，共计150条。数据集包括4个属性，分别为花萼的长、花萼的宽、花瓣的长和花瓣的宽。''')
    elif name == '葡萄酒数据集':
        st.write('''葡萄酒数据集包含在意大利的一个特定区域出产的葡萄酒的化学分析的结
                     果。178个样本中代表了三种葡萄酒，每个样本记录了13种化学分析的结果。''')
    else:
        st.write('''该数据集为威斯康星乳腺癌数据集，总共569个病例，其中212个恶性，357个良性。
                 数据集共有10个基本变量，代表肿瘤图片的病理参数。每个基本变量有三个维度mean, standard error, worst
                 代表某项参数的均值，标准差和最差值，共计是30个特征变量。''')

data_description(dataset_name)



if st.checkbox('查看数据集'):
    st.write('展示数据集的**前5行**')
    st.write(df.head(5))
    st.write('**数据结构:**')
    st.write('注：最后一列为标签列')
    st.info(df.shape)
    st.write('**标签类别:**')
    st.info(T_cn)
    st.write('**描述性统计**')
    st.write(df.describe())
    
    
###train the model

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
score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)
train_predict = clf.predict(X_train)
test_predict = clf.predict(X_test)

    #C = st.sidebar.slider('C', 0.01, 10.0)
    #params['C'] = C
    
def build_model():  
    st.subheader("**2.初步搭建模型**")
    st.write("调整左侧边栏的参数，观察模型准确率，混淆矩阵的变化。")
    st.write('**训练集准确率**: ', round(score_train, 3))
    #st.write([str(i) + str(j) for i, j in zip(T_cn + T_en)])
    cm_columns = ["预测为"+ i for i in T_cn]
    cm_index = ["实际为"+ i for i in T_cn]
    st.table(pd.DataFrame(confusion_matrix(y_train, train_predict),columns = cm_columns,index = cm_index))
    
    st.write('**测试集准确率**: ', round(score_test, 3))
    st.table(pd.DataFrame(confusion_matrix(y_test, test_predict),columns = cm_columns,index = cm_index))    
    st.write("**混淆矩阵解读**：若样本实际为A，预测也为A，则为预测正确样本；若样本实际为A，模型预测为B，则为预测错误样本")
    
    #learning curve
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
    train_sizes = np.linspace(0.1, 1.0, 7)
    
    st.write('**学习率曲线**')
    st.write("通过观察学习曲线了解是否存在过拟合或欠拟合的情况，从而进一步优化模型参数。")
    train_sizes, train_scores, test_scores = learning_curve(clf, X=X, y=y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlabel("sample quantity", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_title("model learning curve", fontsize=12)
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="test score")
    ax.legend(loc="best", fontsize=10)
    st.pyplot(fig)
    st.write("训练一个在训练集和测试集准确率都很高的模型！")
    
build_model()  
##############################################决策树模型####################################    

def Decision_Tree_Visualisation():
    #decision tree plot
    dot_data = tree.export_graphviz(clf
                                    ,feature_names= F_cn
                                    ,class_names= T_cn
                                    ,filled = True
                                    ,rounded= True
                                    ,out_file= None)
    
    st.subheader('**3.数据可视化**')
    st.write('**决策树 ** --当前叶子的个数:',clf.get_n_leaves())
    st.graphviz_chart(dot_data)
    
    ##Feature importance plot
    feature_column = ["Features","Important_Socre"]
    feature_data = pd.DataFrame([*zip(F_en,clf.feature_importances_)],columns =feature_column)
    feature_data = feature_data.sort_values(by = ["Important_Socre"],ascending=False)
    feature_data = feature_data.loc[feature_data['Important_Socre']>0]
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

    ax = sns.barplot(y="Features", x="Important_Socre", data=feature_data)
    fig = ax.get_figure()
    st.pyplot(fig)
    st.write("上述为模型用于分类选择的重要变量，并更具重要性依次排序")
    st.write("**抱歉(￣_,￣ )，由于作图无法显示中文，所以只能用英文变量表示**")
        


def Decision_Boundary_plot():
    ##Decision boundary plot
    
    n_classes = len(np.unique(y))
    color_t = ["red","blue","orange","green"]
    colors = random.sample(color_t, len(np.unique(y)))
    plot_colors = colors
    plot_step = 0.02
    st.subheader('**4.模型修正**--对决策树进行剪枝以防止过拟合')
    st.write('**决策边界图**-- 使用一对分类重要特征')
    st.write('''为了避免过拟合，需要在训练过程中降低决策树的自由度。可以通过设定一些参数来实现。
             最典型的参数是是树的最大深度max_depth，减小树的深度能降低过拟合的风险。
             还有一些其他参数，可以限制决策树的形状：min_sample_split:分裂前节点必须有的最小样本数，
             min_sample_leaf:叶节点必须有的最小样本数量。''')
    image = Image.open('overfitting.jpg')
    st.image(image
             #, caption='Sunrise by the mountains'
             ,use_column_width=True)
    st.write('观察上图可以发现，左图明显为过拟合--训练集决策边界为模型划开过多的边界，降低了模型在测试集的泛化能力。')
    feature_column = ["Features","Important_Socre"]
    feature_data = pd.DataFrame([*zip(F_en,clf.feature_importances_)],columns =feature_column)
    feature_data = feature_data.sort_values(by = ["Important_Socre"],ascending=False)
    feature_data = feature_data.loc[feature_data['Important_Socre']>0]
    feature_B = list(feature_data.iloc[:,0])
    x_B = st.selectbox('选择X轴',feature_B)
    y_B = st.selectbox('选择Y轴',feature_B)
    
    if x_B == y_B:
        st.write("请选择不同的**X**和**Y**来画出决策边界图吧  **!**")
    else:
        fig = plt.figure(figsize=(6,3))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        Xj = pd.DataFrame(X,columns = F_en)
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
                        ,label= get_target_en(dataset_name)[i]
                        ,cmap=plt.cm.PiYG
                        ,edgecolor='black'
                        #,alpha = 0.3
                        ,s=15)
    
        plt.suptitle('Decision boundary plot')
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        st.pyplot(fig)
        
##############################################决策树模型####################################    

Decision_Tree_Visualisation()
Decision_Boundary_plot()
