import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import scipy
from sklearn.model_selection import train_test_split

def create_df(data_path):
    # input: the path of the csv file
    # output: data frame
    df = pd.read_csv(data_path)
    return df

def nan_columns(df):
    # input: data frame
    # output: a list of names of columns that contain nan values in the data frame
    nancolumns = []
    for columns in df.columns:
        if df[columns].isnull().any():
            nancolumns.append(columns)
    return nancolumns

def categorical_columns(df):
    # input: data frame
    # output: a list of column names that contain categorical values in the data frame
    catcolumns = df.select_dtypes(['object']).columns.tolist()
    return catcolumns

def replace_missing_features(df, nancolumns):
    # input: data frame, list of column names that contain nan values
    # output: data frame
    new_df1 = df.copy()
    for columns in nancolumns:
        median = new_df1[columns].median()
        new_df1.update(new_df1[columns].fillna(median))
    return new_df1

def cat_to_num(new_df1, catcolumns):
    # input: data frame, list of categorical feature column names
    # output: data frame
    for i in catcolumns:
        encoder = LabelBinarizer(sparse_output=False)
        cat_column = new_df1[i]
        catcolumn_1hot = encoder.fit_transform(cat_column)
        onehot_df = pd.DataFrame(catcolumn_1hot, columns=encoder.classes_)
        del new_df1[i]
        new_df2 = pd.concat([new_df1, onehot_df], axis=1)
    return new_df2

def standardization(new_df2, labelcol):
    # input: data frame and name of the label column
    # output: scaled data frame
    labelcolumn = new_df2[labelcol]
    scaler = StandardScaler()
    columns_list = list(new_df2.columns)
    new_column_list = columns_list.copy()
    new_column_list.remove(labelcol)
    scaled_values_ndarray = scaler.fit_transform(new_df2[new_column_list])
    scaled_df = pd.DataFrame(scaled_values_ndarray, columns=new_column_list)
    new_df3 = pd.concat([scaled_df,labelcolumn],axis=1)
    return new_df3

def my_train_test_split(new_df3, labelcol, test_ratio):

    np.random.seed(0) # DON'T ERASE THIS LINE
    df_minus_label = new_df3.copy()
    del df_minus_label[labelcol]
    df_label = new_df3[labelcol]
    X_train, X_test = train_test_split(df_minus_label, test_size=test_ratio, random_state = 0)
    y_train, y_test = train_test_split(df_label, test_size = test_ratio, random_state = 0)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    return X_train, X_test, y_train, y_test

def main(dataPath, testRatio, labelColumn):
    # input: the path of the csv file, test data percentage and name of the label column
    # output: X_train, X_test, y_train, y_test as numpy arrays


    df = pd.read_csv(dataPath)

    nancolumns = []
    for columns in df.columns:
        if df[columns].isnull().any():
            nancolumns.append(columns)

    catcolumns = df.select_dtypes(['object']).columns.tolist()

    new_df1 = df.copy()
    for columns in nancolumns:
        median = new_df1[columns].median()
        new_df1.update(new_df1[columns].fillna(median))

    for i in catcolumns:
        encoder = LabelBinarizer(sparse_output=False)
        cat_column = new_df1[i]
        catcolumn_1hot = encoder.fit_transform(cat_column)
        onehot_df = pd.DataFrame(catcolumn_1hot, columns=encoder.classes_)
        del new_df1[i]
        new_df2 = pd.concat([new_df1, onehot_df], axis=1)

    labelcolumn = new_df2[labelColumn]
    scaler = StandardScaler()
    columns_list = list(new_df2.columns)
    new_column_list = columns_list.copy()
    new_column_list.remove(labelColumn)
    scaled_values_ndarray = scaler.fit_transform(new_df2[new_column_list])
    scaled_df = pd.DataFrame(scaled_values_ndarray, columns=new_column_list)
    new_df3 = pd.concat([scaled_df, labelcolumn], axis=1)

    np.random.seed(1)
    df_minus_label = new_df3.copy()
    del df_minus_label[labelColumn]
    df_label = new_df3[labelColumn]
    X_train, X_test = train_test_split(df_minus_label, test_size=testRatio, random_state=0)
    y_train, y_test = train_test_split(df_label, test_size=testRatio, random_state=0)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
    return X_train, X_test, y_train, y_test

"""
X_train, X_test, y_train, y_test = main("housing.csv", 0.3, "median_income")
print("*"*50)
print(y_train.shape)
print("*"*50)
print(y_test.shape)
print("*"*50)
print(X_train)
print("*"*50)
print(X_test.shape)
print("*"*50)
"""