# imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK


# a function for finding information on a feature
def feature_info(feature,values):
    """
    Finds information on a feature.
    
    feature - str, name of a features to find info on
    values - a dataframe with info on variables
    
    Returns known information on requested feature 
    """
    
    return values[values['Attribute']==feature]



# a function to find decoded missing values in the column
def find_nan(df,column,values_nan):
    """
    Finds encoded missing values in the column accordingly
    
    df - dataframe to work with
    column - a name of the column to search for encoded missing values
    values_nan - dataframe with column names and how missing values are encoded
    
    Returns a list where encoded values replaced with NaNs
    """
    
    nan_codes = [int(i) for i in values_nan[values_nan['Attribute']==column]['Value'].values[0].split(',')]
        
    with_nans = [np.nan if value in nan_codes else value for value in df[column]]
    
    return with_nans

# a function to decode missing values back to NaN in the dataset
def decode_nan(df,values_nan):
    """
    Replaces encoded values with NaNs in the dataset
    
    df - a dataset to process
    values_nan - a dataframe with column names and how missing values are encoded
    
    Returns a processed dataset
    """
    
    # replace -1,0 or 9 with NaN if needed
    for col in values_nan['Attribute']:
        
        if col in df.columns:
        
            df[col] = find_nan(df,col,values_nan)
     
    # replace 'XX' with NaN in CAMEO_DEU_2015
    df['CAMEO_DEUG_2015'] = [np.nan if (value=='X') or (value!=value) else int(value) for value in df['CAMEO_DEUG_2015']]
    
    
    return df


# a function to create countplots
def countplot(column,azdias,customers,values_info,figsize=(14,6),show_info=True):
    """
    Shows the difference in people representation of azdias and customers datasets.
    
    column - str, a column name to plot
    azdias - general population information dataframe
    customers - customers information dataframe
    values_info - a dataframe that contains information on columns
    figsize - (float,float), width and height of the plot in inches
    show_info - boolean, if True, shows information on column values
    
    Returns two countplots stacked on each other and info on column values (when show_info=True)
    """
    
    # plot
    fig = plt.figure(figsize=figsize)


    if show_info:

        # information on a column
        value_info = feature_info(column,values_info)
        column_description = value_info['Description'].values[0]
        value_info.drop(['Attribute','Description'], axis=1, inplace=True)
    
        ax1 = fig.add_subplot(121)
        sns.countplot(azdias[column], label='General population', color='red', alpha=0.8, ax=ax1)
        sns.countplot(customers[column], label='Customers', color='green', ax=ax1)
        plt.legend()
    
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        table = ax2.table(cellText=value_info.values, colLabels=value_info.columns,
                         bbox=[0,0,1,1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
    else:
        
        sns.countplot(azdias[column], label='General population', color='red', alpha=0.8)
        sns.countplot(customers[column], label='Customers', color='green')
        plt.legend()

    plt.tight_layout()
    
    return fig

# a function to create a column with peroson's youth decade
def youth_decade(df):
    """
    Extracts information about person's youth decade from PRAEGENDE_JUGENDJAHRE column.
    
    df - a dataframe to extract information from
    
    Retuns a list of numbers that represent a person's youth decade
    (40 if person youth was in 40th, 50 if in 50th and so on)
    """
    
    return df['PRAEGENDE_JUGENDJAHRE'].apply(lambda decade:
                                            40 if decade in [1,2]
                                            else 50 if decade in [3,4]
                                            else 60 if decade in range(5,8)
                                            else 70 if decade in [8,9]
                                            else 80 if decade in range(10,14)
                                            else 90 if decade in [14,15]
                                            else np.nan)

# a function to create a column with peroson's dominating movement in youth
def movement(df):
    """
    Extracts information about person's dominating movement in his/her youth
    
    df - a dataframe to extract information from
    
    Returns a list that represent a person's dominating movement in youth
    """
    
    return df['PRAEGENDE_JUGENDJAHRE'].apply(lambda movement:
                                            'Mainstream' if movement in [1,3,5,8,10,12,14]
                                            else 'Avantgarde' if movement in [2,4,6,7,9,11,13,15]
                                            else np.nan)


# a function to fill in NaNs
def fill_nan(df,values):
    """
    Fills in NaNs in the dataset as follows:
    - round of mean for numerical features
    - modal number for categorical features
    
    df - a dataframe to fill in NaNs
    values - a dataframe with information on variables
    
    Returns a dataset without NaNs
    """

    # numerical variables
    numeric = [col for col in values[values['Meaning'].str.contains('numeric', case=False, na=False)]['Attribute'] \
           if col in df.columns]


    # categorical variables
    categorical = [col for col in df.columns if col not in numeric]
    
    for col in df[numeric].columns:
        df[col].fillna(round(df[col].mean()), inplace=True)
        
    for col in df[categorical].columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
        
    return df


# a function to label object type data
def label_obj(df):
    """
    Uses sklearn's LabelEncoder to label object type data
    
    df - a dataframe to label data
    
    Returns a dataset with labeled object type data
    """
    
    for col in df.select_dtypes(['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
        
        
    return df


# a function to scale the data
def scale_data(df):
    """
    Uses sklearn's StandardScaler to scale the data
    
    df - dataframe with data to be scaled
    
    Returns a scaled dataset
    """
    
    df_scaled = pd.DataFrame(data=StandardScaler().fit_transform(df),
                            columns=df.columns)
    
    return df_scaled


# a function for applying PCA
def apply_pca(df_scaled,num):
    """
    Apllies PCA to the data

    df_scaled - a scaled data for PCA to fit and transform
    num - int, number of principal components

    Returns a dataframe with reduced number of dimmensions
    """

    df_pca = pd.DataFrame(data=PCA(n_components=num, random_state=42).fit_transform(df_scaled),
                        columns=[f'pc_{i}' for i in range(1,num+1)])
  
    return df_pca

# a fucntion to find an optimal k
def find_k(df,clusters=range(2,21)):
    
    """
    
    Uses KMeans
    df - a dataframe to work on,
    clusters - list or array of number of clusters to go through
    
    Returns a dataframe with number of cluster and sum of squared distances as columns sorted
    by the distances
    """
    
    # initiate lists to populate
    num_k, ssd = [],[]
    
    
    # go through clusters to find the best one
    for k in tqdm(clusters):
        
        num_k.append(k)
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        
        kmeans.fit(df)
        
        ssd.append(kmeans.inertia_)
        
    
    scores_df = pd.DataFrame(data={
        'Number of clusters':num_k,
        'Sum of squared distances':ssd
    })
    
    
    return scores_df

# a class object to process the data
class ProcessForClustering(BaseEstimator,TransformerMixin):
    """
    Creates an object to process the data before clustering doing the following:
    - picks out the columns to work with
    - drops duolicated rows
    - decodes missing values back to NaNs
    - engineers new columns and drops useless ones
    - fills in NaNs (mean value for numeric, modal value for categorical variables)
    - labels object type data

    In order for the object to work properly the following variables have to be created 
    before using the class:

    values - a dataframe with information on the variables
    values_nan - a dataframe with information on how missing values are coded in the dataframes
    final_cols - list of columns to work with while processing the data
    """
    
    def __init__(self,values_nan,values,fill_na=True,label_obj=True):
        
        """
        Initializes the object.
        
        values_nan - a dataframe with information on how missing values are coded in dataframe
        fill_na - boolean, True to fill in missing values
        label_obj - boolean, True to label object type data
        """
        
        self.values = values
        self.values_nan = values_nan
        self.fill_na = fill_na
        self.label_obj = label_obj
    
    def fit(self,X,y=None):
        
        return self
    
    def transform(self,X,y=None):
        
        self.X = X

        with open('final_columns.pickle','rb') as f:
            final_cols = pickle.load(f)

        X_proc = self.X.drop(['Unnamed: 0','LNR','CAMEO_DEU_2015'], axis=1)

        X_proc.drop_duplicates(inplace=True)
        
        X_proc = X_proc[final_cols]
        
        X_proc = decode_nan(X_proc,self.values_nan)
        
        X_proc['YOUTH_DECADE'] = youth_decade(X_proc)
        
        X_proc['MOVEMENT'] = movement(X_proc)
        
        X_proc.drop('PRAEGENDE_JUGENDJAHRE', axis=1, inplace=True)
        
        if self.fill_na:
        
            X_proc = fill_nan(X_proc,self.values)
            
        if self.label_obj:
        
            X_proc = label_obj(X_proc)
        
        return X_proc


# function to visually compare general population to customer clusters
def compare(general,customer,column,values_info,figsize=(14,6),polar=True):
    """
    Visually compares general population information to customer clusters 6,9 and 4
    
    general - a dataframe with general population information
    customer - a dataframe with customers' information
    column - str, name of the column to compare
    values_info - a dataframe that contains infromation on a column
    figsize - (float,float), width and height of the figure
    polar - boolean, if True plots a radar chart, if False creates a coutplot

    In order for function to work, feature_info() function (show info on a variables) has to be defined.
    
    Returns a radar chart or countplot on left side and column descriprion on right side.
    """
    
    # variables for the plot
    
    # categories and their share in general and customer clusters' population
    categories = list(general[column].value_counts().index)
    general_share = list(round(general[column].value_counts()/len(general)*100,1).values)
    cluster6_share = [round(customer[customer['CLUSTERS']==6][column].value_counts()/len(customer[customer['CLUSTERS']==6])\
             *100,1)[i] for i in categories]
    cluster9_share = [round(customer[customer['CLUSTERS']==9][column].value_counts()/len(customer[customer['CLUSTERS']==9])\
             *100,1)[i] for i in categories]
    cluster4_share = [round(customer[customer['CLUSTERS']==4][column].value_counts()/len(customer[customer['CLUSTERS']==4])\
             *100,1)[i] for i in categories]
    # splitting the radar chart
    angles = np.linspace(0,2*np.pi,len(categories), endpoint=False)
    
    # information on a column
    value_info = feature_info(column,values_info)
    column_description = value_info['Description'].values[0]
    value_info.drop(['Attribute','Description'], axis=1, inplace=True)

    
    
    # plotting the graph
    fig = plt.figure(figsize=figsize)
    
    
    if polar:
        
        # full circle
        angles = np.concatenate((angles,[angles[0]]))
        
        for variable in [categories,general_share,cluster6_share,cluster9_share,cluster4_share]:
            variable.append(variable[0])
    
    
        # radar chart
        ax1 = fig.add_subplot(121,polar=True)
        # plot general population 
        ax1.plot(angles,general_share, 'o-', lw=1, color='green', label='General population')
        ax1.fill(angles,general_share, color='green', alpha=0.25)
        # plot cluster 6
        ax1.plot(angles,cluster6_share, 'o-', lw=1, color='red', label='Customer cluster 6')
        ax1.fill(angles,cluster6_share, color='red', alpha=0.25)
        # plot cluster 9
        ax1.plot(angles,cluster9_share, 'o-', lw=1, color='black', label='Customer cluster 9')
        ax1.fill(angles,cluster9_share, color='black', alpha=0.25)
        # plot cluster 4
        ax1.plot(angles,cluster4_share, 'o-', lw=1, color='purple', label='Customer cluster 4')
        ax1.fill(angles,cluster4_share, color='purple', alpha=0.25)
        
        # adding labels
        ax1.set_thetagrids(angles*180/np.pi,categories)
        plt.legend()
        
    else:
        
        # plot a barplot        
        df = pd.DataFrame(data={'Categories':categories,
                                'General population':general_share,
                                'Customer cluster 6':cluster6_share,
                                'Customer cluster 9':cluster9_share,
                                'Customer cluster 4':cluster4_share})

        ax1 = fig.add_subplot(121)
        
        df.plot(x='Categories', y=['General population','Customer cluster 6','Customer cluster 9','Customer cluster 4'],
               kind='bar', rot=0, ax=ax1)

    
    # text on the right side
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    column_info = ax2.table(cellText=value_info.values, colLabels=value_info.columns,
                           bbox=[0,0,1,1])
    column_info.auto_set_font_size(False)
    column_info.set_fontsize(10)
    
    # column description
    fig.suptitle(column_description)
    
    plt.tight_layout()
    
    return fig



# a function to decode missing values back to NaN in the dataset
def classification_decode_nan(df,values_nan):
    """
    Replaces encoded values with NaNs in the dataset for classifictaion problem
    
    df - a dataset to process
    values_nan - a dataframe with column names and how missing values are encoded
    
    Returns a processed dataset
    """
    
    # replace -1,0 or 9 with NaN if needed
    for col in values_nan['Attribute']:
        
        if col in df.columns:
        
            df[col] = find_nan(df,col,values_nan)
     
    # replace 'XX' with NaN in CAMEO_DEU_2015
    df['CAMEO_DEUG_2015'] = [np.nan if (value=='X') or (value!=value) else int(value) for value in df['CAMEO_DEUG_2015']]
    df['CAMEO_DEU_2015'] = [np.nan if (value=='XX') or (value!=value) else value for value in df['CAMEO_DEU_2015']]
    df['CAMEO_INTL_2015'] = [np.nan if (value=='XX') or (value!=value) else int(value) for value in df['CAMEO_INTL_2015']]
    
    
    return df


# a function to engineer new columns
def new_columns(df):
    """
    Engineers new columns and drops the columns that were used to create new ones
    
    df - a dataframe to work with

    In order for function to work, youth_decade() and movement() functions
    have to be defined
    
    Returns a dataframe with some new columns
    """
    
    # keep year out of 'EINGEFUEGT_AM_YEAR' column
    df['EINGEFUEGT_AM_YEAR'] = pd.to_datetime(df['EINGEFUEGT_AM']).dt.year
    df['CAMEO_DEU_2015_1'] = [val if val!=val else int(val[0]) for val in df['CAMEO_DEU_2015']]
    df['CAMEO_DEU_2015_2'] = [val if val!=val else val[-1] for val in df['CAMEO_DEU_2015']]
    df['YOUTH_DECADE'] = youth_decade(df)
    df['MOVEMENT'] = movement(df)
    
    df.drop(['EINGEFUEGT_AM','CAMEO_DEU_2015','PRAEGENDE_JUGENDJAHRE'],
           axis=1, inplace=True)
    
    return df



# a function to fill in NaNs
def classification_fill_nan(df,values):
    """
    Fills in NaNs in the dataset as follows:
    - round of mean for numeric features
    - modal number for categorical features
    
    df - a dataframe to fill in NaNs
    values - a dataframe with information on variables
    
    Returns a dataset without NaNs
    """

    # numerical variables
    numeric = [col for col in values[values['Meaning'].str.contains('numeric', case=False, na=False)]['Attribute'] \
           if col in df.columns]
    numeric.append('EINGEFUEGT_AM_YEAR')


    # categorical variables
    categorical = [col for col in df.columns if col not in numeric]
    
    for col in df[numeric].columns:
        df[col].fillna(round(df[col].mean()), inplace=True)
        
    for col in df[categorical].columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
        
    return df


# function to create an undersampled dataset
def undersample(df,sklearn_shuffle,seed=42):
    """
    Creates an undersampled dataset out of the existing one
    
    df - a dataframe to undersample
    sklearn_shuffle - sklearn.utils shuffle module
    seed - int, to determine randomness
    
    Returns a dataframe with even number of positive and negative responses
    """
    
    positive = df[df['RESPONSE']==1]
    negative = df[df['RESPONSE']==0].sample(n=len(positive), random_state=seed)
    
    undersampled = pd.concat([positive,negative])
    undersampled = sklearn_shuffle(undersampled)
    
    return undersampled


# an object class to process the data
# a class object to process the data
class ProcessForClassification(BaseEstimator,TransformerMixin):
    """
    Creates an object to process the data before classification doing the following:
    - picks out the columns to work with
    - decodes missing values back to NaNs
    - engineers new columns and drops useless ones
    - fills in NaNs (mean value for numeric, modal value for categorical variables)
    - labels object type data

    In order for the object to work properly the following variables have to be created 
    before using:

    values - a dataframe with information on the variables
    values_nan - a dataframe with information on how missing values are coded in the dataframes
    feature_importance - a Series that contains information on how important a variable for
                        Random Forest decision making
    classification_cols.pkl - pickled file that contains list of columns to work with while processing the data
    """
    
    def __init__(self,values_nan,values,feature_importance,threshold=0.002):
        
        """
        Initializes the object.
        
        values_nan - a dataframe with information on how missing values are coded in dataframe
        values - a dataframe with information on variables
        feature_importance - a Series that contains information on how important a variable for
                            Random Forest decision making
        threshold - float, a threshold to use to filter out variables
        """
        
        self.values = values
        self.values_nan = values_nan
        self.feature_importance = feature_importance
        self.threshold = threshold
    
    def fit(self,X,y=None):
        
        return self
    
    def transform(self,X,y=None):
        
        self.X = X

        with open('classification_cols.pkl','rb') as f:
            keep_cols = pickle.load(f)
            
        # important variables
        important_cols = self.feature_importance[self.feature_importance>self.threshold].index.to_list()
        
        X_proc = self.X[keep_cols]
        
        # decode missing values
        X_proc = classification_decode_nan(X_proc,self.values_nan)
        
        # engineer new columns
        X_proc = new_columns(X_proc)
        
        # fill in NaNs
        X_proc = classification_fill_nan(X_proc,self.values)
        
        # label object type data
        X_proc = label_obj(X_proc)
        
        X_proc = X_proc[important_cols]
        
        return X_proc


# a function to train and evaluate classifiers
def train_model(X,y,models,values_nan,values,feature_importance,threshold=0.002,cross_validation=cross_val_score,
                split=StratifiedShuffleSplit):
    """
    Uses cross validation to train different pipelines.
    Evaluates models using ROC_AUC score

    models - a dictionary of classifiers to try in pipeline
    X - features
    y - target
    values_nan - a dataframe with information on how missing values are coded in dataframe
    values - a dataframe with information on variables
    feature_importance - a Series that contains information on how important a variable for
                            Random Forest decision making
    threshold - float, a threshold to use to filter out variables 
    cross_validation - sklearn's cross_val_score module
    split - sklearn's StratifiedShuffleSplit module

    In order to work an object ProcessForClassification and functions it's using
    have to be defined.

    Returns a dataframe where classifiers are sorted by their ROC_AUC score
    """
    
    model_name,model_score = [],[]
    
    for name,model in tqdm(models.items(), desc='Training...'):
        
        model_name.append(name)
        
        # pipeline
        pipeline = Pipeline(steps=[
            ('features',ProcessForClassification(values_nan,values,feature_importance,threshold)),
            ('clf',model)
        ])
        
        # 10 fold shuffle split
        split = StratifiedShuffleSplit(n_splits=10)
        
        # cross validation
        mean_score = cross_val_score(pipeline,X,y,cv=split,scoring='roc_auc',n_jobs=-1, error_score='raise').mean()
        
        model_score.append(mean_score)
        
    scores_df = pd.DataFrame(data={'Classifier':model_name,
                                   'ROC AUC Score':model_score})
    
    scores_df.sort_values(by='ROC AUC Score', ascending=False, inplace=True)
    
    return scores_df


# function to prepare a file for submission
def file_for_submission(clf,X,y,test,file_name,values,values_nan,important_features,threshold=0.002):
    """
    Builds a pipeline to process data, trains it and make predictions for the test set,
    Saves the result into .csv file to be submitted on kaggle.com
    
    clf - classifier with tuned parameters to use in pipeline
    X - train set features
    y - train set target
    test - data to predict on
    file_name - str, name of a submission file to use (has to be .csv format)
    values - dataframe with info on features
    values_nan - dataframe with info in missing values
    important_features - list of features to work with
    threshold - float, threshold for selecting features

    In order to work an object ProcessForClassification and functions it's using
    have to be defined.

    Returns a dataframe of positive probabilities and saves it into .csv file
    ready to be submitted on kaggle.com
    """
    
    pipeline = Pipeline(steps=[
        ('features',ProcessForClassification(values_nan,values,important_features,threshold)),
        ('clf',clf)
    ])
    
    # train classifier
    pipeline.fit(X,y)
    
    # make predictions / save only probabilities of a positive response
    proba = pipeline.predict_proba(test)[:,1]
    
    # a dataframe with data to submit
    submission = pd.DataFrame({'LNR':test['LNR'],
                               'RESPONSE':proba})
    
    # save into .csv file
    submission.to_csv(file_name, index=False)
    
    return submission