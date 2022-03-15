import math
from math import sqrt

import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from witwidget.notebook.visualization import WitWidget, WitConfigBuilder



from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


from sklearn.metrics import accuracy_score
import pandas as pd
import shap

st.set_page_config(
    page_title="Auto Machine Learning Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎈",

)


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)


hide_streamlit_style = """
<style>
.css-hi6a2p {padding-top: 0rem;}
</style>

"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# st.title('Demo for Interactive ML ')

# st.title("""
#  Visually Explore Machine Learning Prediction
# """)

from enum import Enum
from io import BytesIO, StringIO
from typing import Union

import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer







FILE_TYPES = ["csv",'xlsx']


st.write("""
**Task Description**

You will see a dataset provided by the University of Wisconsin. There are 30 numeric variables that show the measurements of the needle aspirate for each patient's breast mass, one variable indicating each patient's identification number, and one variable indicating the diagnosis(1 = malignant, 0 = benign).
You will be asked to predict whether the tumor is benign or malignant based on some diagnosed cases' information. You can make your own predictions, or choose to accept or reject the machine learning algorithm's predictions. 


""")

@st.cache(allow_output_mutation=True)
def loaddata():
    data=load_breast_cancer()
    df= pd.DataFrame(data.data, columns=data.feature_names)
    df['Diagnosis'] = data.target
    return df

with st.expander("Dataset Preview"):

    df = loaddata()

    st.dataframe(df)

# data = load_breast_cancer()
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['Diagnosis'] = data.target

@st.cache(allow_output_mutation=True)
def dfinfo():
    info=df.describe()
    return info

with st.expander("More information about the variables"):
    st.write('''
Attribute Information:\n
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)
'''
)
    if st.checkbox('Show Dataset Summary'):
        st.write(dfinfo())


with st.expander("Dataset Processing"):
    col1, col2,col3 = st.columns(3)
    with col1:
        if st.checkbox('Show Missing Values'):
            percent_missing = df.isnull().sum() * 100 / len(df)
            missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
            st.dataframe(missing_value_df)

    with col2:
        if st.checkbox('Show Duplication'):
            duplicated = df.duplicated().sum()
            dupratio = round(duplicated / len(df), 2)
            st.write("Duplication rate:", dupratio)
            if dupratio != 0:
                duplicateRowsDF = df[df.duplicated()]
                st.dataframe(duplicateRowsDF)
    with col3:
        if st.checkbox('Show Correlation'):
            bound=st.selectbox('Select the minimal correlated rate to check',(0.9,0.8,0.7))
            fig3,ax= plt.subplots()

            xCorr = df.corr()
            kot = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr !=1.000)]
            sns.heatmap(kot, cmap="Greens")
            st.write(fig3)


        # def corrFilter(x: pd.DataFrame, bound: float):
        #     xCorr = x.corr()
        #     xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr != 1.000)]
        #     xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
        #     return xFlattened






with st.expander("Dataset Exploration"):
    col1, col2  = st.columns(2)

    with col1:
        st.subheader('Histogram for each feature')

        feature = st.selectbox('Choose the feature', ['mean radius','mean texture', 'mean perimeter' ,'mean area',
 'mean smoothness' ,'mean compactness' ,'mean concavity',
 'mean concave points', 'mean symmetry' ,'mean fractal dimension',
 'radius error', 'texture error' ,'perimeter error' ,'area error',
 'smoothness error', 'compactness error' ,'concavity error',
 'concave points error', 'symmetry error' ,'fractal dimension error',
 'worst radius', 'worst texture' ,'worst perimeter' ,'worst area',
 'worst smoothness' ,'worst compactness', 'worst concavity',
 'worst concave points' ,'worst symmetry', 'worst fractal dimension'])
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        st.write('\n')

        fig1 = px.histogram(df, x=feature, color="Diagnosis", marginal="rug")
        st.plotly_chart(fig1,use_container_width=True)


    with col2:
        st.subheader('Scatter plot for variable interaction')

        X_Value = st.selectbox(
        'Select X-axis value',
        ('mean radius','mean texture', 'mean perimeter' ,'mean area',
 'mean smoothness' ,'mean compactness' ,'mean concavity',
 'mean concave points', 'mean symmetry' ,'mean fractal dimension',
 'radius error', 'texture error' ,'perimeter error' ,'area error',
 'smoothness error', 'compactness error' ,'concavity error',
 'concave points error', 'symmetry error' ,'fractal dimension error',
 'worst radius', 'worst texture' ,'worst perimeter' ,'worst area',
 'worst smoothness' ,'worst compactness', 'worst concavity',
 'worst concave points' ,'worst symmetry', 'worst fractal dimension')
    )

        Y_Value = st.selectbox(
        'Select Y-axis value',
        ('mean radius','mean texture', 'mean perimeter' ,'mean area',
 'mean smoothness' ,'mean compactness' ,'mean concavity',
 'mean concave points', 'mean symmetry' ,'mean fractal dimension',
 'radius error', 'texture error' ,'perimeter error' ,'area error',
 'smoothness error', 'compactness error' ,'concavity error',
 'concave points error', 'symmetry error' ,'fractal dimension error',
 'worst radius', 'worst texture' ,'worst perimeter' ,'worst area',
 'worst smoothness' ,'worst compactness', 'worst concavity',
 'worst concave points' ,'worst symmetry', 'worst fractal dimension')
    )



    # with st.echo(code_location='below'):
    #     import plotly.express as px

        fig2 = px.scatter(df,
                x=df[X_Value],
                y=df[Y_Value],
                color="Diagnosis"
            )
        fig2.update_layout(
                xaxis_title=X_Value,
                yaxis_title=Y_Value,
            )

        st.plotly_chart(fig2, use_container_width=True)



y=df.iloc[: ,-1]
X = df.iloc[: , 0:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

with st.expander("Diagnose New Patients"):
    NewPatients = st.selectbox(
        'Choose the patient id in the test set', [i for i in range(1,len(X_test))],key=1)
    st.dataframe(X_test.iloc[[NewPatients-1]])

    Diagnose = st.selectbox(
        'Choose your diagnosis: The tumor of this patient is ', ('', 'malignant', 'benign'))

    rightanswer = y_test.iloc[NewPatients - 1]

    if rightanswer == 1:
        rightanswer = 'malignant'
    else:
        rightanswer = 'benign'

    check = 'do not know'
    if rightanswer == Diagnose:
        check = 'right'
    else:
        check = 'wrong'

    if Diagnose !='':
        st.write(
            'You diagnosed that the tumor of this patient is %s, in fact the tumor of this patient is %s. You got the %s diagnosis!' % (
            Diagnose, rightanswer, check))
        st.write(
            'You can also use some amazing machine learning algorithms to help your diagnosis!')

from sklearn.ensemble import RandomForestClassifier          # 随机森林
from sklearn.svm import SVC, LinearSVC                       # 支持向量机
from sklearn.linear_model import LogisticRegression          # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier           # KNN算法
from sklearn.naive_bayes import GaussianNB                   # 朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier              # 决策树分类器
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingClassifier

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@st.cache(suppress_st_warning=True)
def checkdia(y_pred,patientsid):
    rightanswer = y_test.iloc[NewPatients - 1]
    predvalue=y_pred[NewPatients - 1]

    if rightanswer == 1:
        rightanswer = 'malignant'
    else:
        rightanswer = 'benign'

    if predvalue ==1:
        predvalue = 'malignant'
    else:
        predvalue = 'benign'

    check = 'do not know'
    if rightanswer == predvalue:
        check = 'right'
    else:
        check = 'wrong'

    if predvalue!='':
        st.write(
            'The model diagnosed that the tumor of this patient is %s, in fact the tumor of this patient is %s. The model got the %s diagnosis!' % (
            predvalue , rightanswer, check))







def show_perf_metrics(y_test, pred):
    """show model performance metrics such as classification report or confusion matrix"""
    target_name=df.iloc[:,-1].unique()
    # report = classification_report(y_test, pred, target_names= target_name, output_dict=True)
    # st.dataframe(pd.DataFrame(report).round(1).transpose())
    conf_matrix = confusion_matrix(y_test, pred,list(set(y_test)),labels= target_name)
    sns.set(font_scale=1.4)
    ax = plt.subplot()
    sns.heatmap(
        conf_matrix,
        square=True,
        annot=True,
        annot_kws={"size": 15},
        cmap="YlGnBu",

        cbar=False,
    )
    plt.xlabel('Prediction Class')
    plt.ylabel('Actual Class')
    ax.xaxis.set_ticklabels(target_name)
    ax.yaxis.set_ticklabels(target_name)

    st.pyplot(use_container_width=True)
    st.markdown('There are %d maligent tutmors, the models diagnosed %d of them correctly. The accuacry is %d percent. '
                %((conf_matrix[1][0]+conf_matrix[1][1]),conf_matrix[1][1], (conf_matrix[1][1]/ (conf_matrix[1][0]+conf_matrix[1][1]) )*100  ))
    st.markdown('There are %d benign tutmors, the models diagnosed %d of them correctly. The accuacry is %d percent. '
                % ((conf_matrix[0][0] + conf_matrix[0][1]), conf_matrix[0][0],(conf_matrix[0][0]/ (conf_matrix[0][0]+conf_matrix[0][1]) )*100 ))


Classifiers=[["Random Forest",RandomForestClassifier()],
                     # ["Support Vector Machine",SVC()],
                     ["Logistic Regression",LogisticRegression()],
                     ["KNN", KNeighborsClassifier(n_neighbors=5)]]

params =[{ 'n_estimators': [20, 30, 40, 50],
                   'max_depth': [5, 10, 15],
                   'max_features': [2, 3],
                   'min_samples_leaf': [5, 10],
                   'min_samples_split': [10,20]


        },
         {
            'penalty': ['l1', 'l2'],
            'C': [0.01,0.09, 1, 5] },

dict(n_neighbors=list(range(3, 15)))

]

import numpy as np
np.random.seed(1234)


@st.cache(show_spinner=False,allow_output_mutation=True)

def findthebestmodel():
    gridmodel0 = GridSearchCV(Classifiers[0][1], params[0], cv=5, scoring='accuracy', return_train_score=False)
    gridmodel0.fit(X_train, y_train)


    gridmodel1 = GridSearchCV(Classifiers[1][1], params[1], cv=5, scoring='accuracy', return_train_score=False)
    gridmodel1.fit(X_train, y_train)


    gridmodel2 = GridSearchCV(Classifiers[2][1], params[2], cv=5, scoring='accuracy', return_train_score=False)
    gridmodel2.fit(X_train, y_train)


    return gridmodel0, gridmodel1,gridmodel2

models=findthebestmodel()

@st.cache(show_spinner=False,allow_output_mutation=True)
def modelpred():

    y_pred0 = models[0].predict(X_test)
    y_pred1 = models[1].predict(X_test)
    y_pred2 = models[2].predict(X_test)
    return y_pred0,y_pred1,y_pred2

modelpreds=modelpred()

def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)



    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df = fi_df.head(5)
    st.markdown(
        'The model thinks the top 5 most informative variables when it makes predictions are: %s' % fi_df['feature_names'].unique())

    #Define size of bar plot
    # plt.figure(figsize=(10,8))
    # #Plot Searborn bar chart
    # sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # #Add chart labels
    # plt.title(model_type + ' FEATURE IMPORTANCE')
    # plt.xlabel('FEATURE IMPORTANCE')
    # plt.ylabel('FEATURE NAMES')
    # st.pyplot()




with st.expander("Train Machine learning Models"):
    start = st.checkbox('Start training Machine learning models')

    if start:
        st.write('Three models will be trained on the training data, their performance will be shown. You can choose the model to make the prediction.')
        col1, col2,col3= st.columns(3)

        with col1:
            st.write(Classifiers[0][0])
            show_perf_metrics(y_test, modelpreds[0])
            plot_feature_importance(models[0].best_estimator_.feature_importances_, X.columns, 'Random Forest')


        with col2:
            st.write(Classifiers[1][0])
            show_perf_metrics(y_test, modelpreds[1])
            plot_feature_importance(models[1].best_estimator_.coef_[0], X.columns, 'KNN')


        with col3:
            st.write(Classifiers[2][0])
            show_perf_metrics(y_test, modelpreds[2])

            st.markdown('This model finds the %d most similar cases to make the prediction' % models[2].best_params_['n_neighbors'])


def trans(ip):
    result =''
    if ip == 1:
        result ='maligent'
    else:
        result = 'benign'
    return result



import shap  # package used to calculate Shap values
import streamlit.components.v1 as components

def shapexp(model, patientid):
    model = models[0].best_estimator_
    explainer = shap.TreeExplainer(model)
    data_for_prediction = X_test.iloc[patientid - 1]

    shap_values = explainer.shap_values(data_for_prediction)

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction))

def shapexplg(model, patientid):
    model = models[1].best_estimator_
    explainer = shap.LinearExplainer(model,X_train)

    data_for_prediction = X_test.iloc[patientid - 1]

    shap_values = explainer.shap_values(data_for_prediction)

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    st_shap(shap.force_plot(explainer.expected_value, shap_values,feature_names=X.columns))


def shapexpknn(model, patientid):
    model = models[2].best_estimator_
    explainer = shap.KernelExplainer(model.predict_proba, X_train)

    # Get shap values for the test data observation whose index is 0, i.e. first observation in the test set

    data_for_prediction = X_test.iloc[patientid - 1]

    shap_values = explainer.shap_values(data_for_prediction)


    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[1], X_test.iloc[0, :]))


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for i in range(len(train)):

        dist = euclidean_distance(test_row, train.iloc[i])
        distances.append((train.iloc[i], dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    knn = pd.DataFrame(neighbors)
    index = knn.index
    print(index)
    a = y_train[y_train.index.isin(index)]
    knn['Diagnosis'] = a
    return knn



with st.expander("Machine learning Predictions"):
    st.markdown('The patient with id number: %d in the test set'%(NewPatients))
    col1, col2, col3 = st.columns(3)
    with col1:

        rfpred = modelpreds[0][NewPatients-1]

        res= trans(rfpred)
        st.markdown('Random Forest model predicts its tumor as %s.' % res)
        shapexp(models[0], NewPatients)



    with col2:
        logpred = modelpreds[1][NewPatients-1]
        res= trans(logpred)
        st.markdown('Logistic regression model predicts its tumor as %s.' % res)
        shapexplg(models[1], NewPatients)

    with col3:
      knnpred = modelpreds[2][NewPatients-1]
      res = trans(knnpred)
      st.markdown('KNN model predicts its tumor as %s.' % res)
      shapexpknn(models[2], NewPatients)
      neighbors = get_neighbors(X_train, X_test.iloc[NewPatients-1], 4)
      st.write('The most similar cases are:')
      st.dataframe(neighbors)















        # with col4:
        #     st.write(Classifiers[1][0])
        #     gridsvm = GridSearchCV(Classifiers[1][1], paramsvm, cv=5, scoring='accuracy', return_train_score=False)
        #     gridsvm.fit(X_train, y_train)
        #     y_pred = gridsvm.predict(X_test)
        #     show_perf_metrics(y_test, y_pred)


























# def get_dataset(upload_file,name):
#     if upload_file is not None:
#         df= pd.read_csv(upload_file, encoding="utf8")
#         # replace all non alphanumeric column names to avoid lgbm issue
#         df.columns = [
#             "".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns
#         ]
#     if name == 'Iris':
#         data = datasets.load_iris()
#
#     else:
#         data = datasets.load_wine()
#
#     X = data.data
#     y = data.target
#     return X, y





# st.set_option('deprecation.showfileUploaderEncoding', False)


#
# # def datapro(name):
#
# def getdf():
#
#          data = datasets.load_iris()
#          df = pd.DataFrame(data.data, columns=['Sepal Length',
#                                          'Sepal Width',
#                                          'Petal Length',
#                                           'Petal Width'])
#          df['Class'] = 'Setosa'
#          df.loc[50:100, 'Class'] = 'Versicolor'
#          df.loc[100:150, 'Class'] = 'Virginica'
#          # if st.checkbox('Show df'):
#          #     st.write(df)
#          return df
#
# df=getdf()
#
# st.write('Shape of dataset:', df.shape)
#
        # paramsvm = {'C': [0.1, 1, 10],
        #             'gamma': [1, 0.1, 0.01],
        #             'kernel': ['rbf', 'poly', 'sigmoid']}

