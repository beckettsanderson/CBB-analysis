# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:42:09 2022

@author: Beckett Sanderson
         Benoit Cambournac
         Julian Savini
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

BASIC = "CBB Basic Stats.txt"
ADVANCED = "CBB Advanced Stats.txt"

def read_csv(filename, headers = None):
    """
    Reads in a csv with pandas to create a data frame

    Parameters
    ----------
    filename : string
        the location of the file to read in data from.
    headers : list
        the names for the headers of the columns

    Returns
    -------
    df : data frame
        a data frame containing all the data from the csv.

    """
    df = pd.read_csv(filename)
    
    # sets column headers if there are none already
    if headers != None:
        
        df.columns = headers
    
    print(df.head(), "\n")
    print(df.tail(), "\n")
    print("Number of Instances:", len(df), "\n")
    
    return df


def merge_and_clean(bas_df, adv_df):
    """
    Merge together our two dataframes, clean them, and prepare for analysis

    Parameters
    ----------
    bas_df : data frame
        data frame containing all the basic stats by team.
    adv_df : data frame
        data frame containing all the advanced stats by team.

    Returns
    -------
    new_df : data frame
        data frame containing all the stats by team and if they made 
        the tournament.

    """
    # merge the two data frames
    new_df = bas_df.merge(adv_df)
    
    # create the tourney column of the data frame
    new_df["Tourney"] = None
    
    # create NCAA tourney column where Y if in tournament and N if not
    for ind in new_df.index:
        
        # check if the school participated in the NCAA tournament
        if new_df["School"][ind].endswith("NCAA"):
            
            # set the tourney column to Y for true and remove NCAA from name
            new_df["Tourney"][ind] = 1
            new_df["School"][ind] = new_df["School"][ind].rstrip("NCAA").strip()
        
        else:
            
            # set the tourney column to N for false
            new_df["Tourney"][ind] = 0
    
    # show results of merge and clean
    print(new_df.head())
    
    return new_df
    
    
def joinplot(df, xcol, ycol, xlab, ylab, title, color="m"):
    """
    Plots a joinplot of two different statistics for all teams

    Parameters
    ----------
    df : data frame
        data frame containing all the data for different teams.
    xcol : string
        the x data column.
    ycol : string
        the y data column.
    xlab : string
        x axis label.
    ylab : string
        y axis label.
    title : string
        title of the graph.
    color : string
        color to plot the graph.

    Returns
    -------
    None.

    """
    sns.set_theme(style="darkgrid")
    j = sns.jointplot(x = xcol, y = ycol, data=df, color=color, kind="reg")
    
    # graph organization
    j.fig.suptitle(title, y=1.05)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    
    
def plot_joinplots(all_df):
    """
    Plota section of joinplots or particular interest to us

    Parameters
    ----------
    all_df : data frame
        data frame with all the college data for each team.

    Returns
    -------
    None.

    """
    joinplot(all_df, "W-L%", "TS%", "Win/Loss Percentage", 
                "True Shooting Percentage", 
                "Win/Loss Percentage vs True Shooting Percentage")
    joinplot(all_df, "W-L%", "TRB%", "Win/Loss Percentage", 
                "Total Rebounding Percentage", 
                "Win/Loss Percentage vs Total Rebounding Percentage")
    joinplot(all_df, "FGA", "ORB%", "Field Goal Attempts", 
                "Offensive Rebounding Percentage", 
                "Field Goal Attempts vs Offensive Rebounding Percentage")
    joinplot(all_df, "W-L%", "TOV%", "Win/Loss Percentage", 
                "Turnover Percentage", 
                "Win/Loss Percentage vs Turnover Percentage")
    joinplot(all_df, "W-L%", "BLK%", "Win/Loss Percentage", 
                "Block Percentage", 
                "Win/Loss Percentage vs Block Percentage")
    

def boxplot(df, ycol, palette=["m", "g"]):
    """
    Plots a boxplot of a statistic compared to making the NCAA tournament 
    for all teams

    Parameters
    ----------
    df : data frame
        data frame containing all the data for different teams.
    ycol : string
        the column to compare to making the tournament

    Returns
    -------
    None.

    """
    # set theme of the plot
    sns.set_theme(style="ticks", palette="pastel")
    
    # draw a boxplot for making and not making the NCAA tournament
    sns.boxplot(x="Tourney", y=ycol, palette=palette, data=df)
    
    # graph organization
    plt.title(ycol + " and Making the NCAA Tourney")
    plt.xlabel("Make the NCAA Tournament")
    plt.ylabel(ycol)
    plt.show()
    

def split_data(df):
    """
    
    Split data from combined dataframe into testing and training data.
    
    Parameters
    ----------
    df : dataframe
        A combined dataframe of the basic and advanced statistics

    Returns
    -------
    X_train : dataframe
        The values of the dataframe
    X_test : dataframe
        The split testing data
    y_train : series
        The feature which corresponds to the input values in X_train
    y_test : series
        The feature which corresponds to the input values from X_test

    """
    # for machine learning aspects, set Tourney to "Yes" and "No"
    for ind in df.index:
        
        # check if the school made the tournament
        if df["Tourney"][ind] == 1:
            # set the tourney column to Yes if they made it
            df["Tourney"][ind] = "Yes"
        
        else:
            # set the tourney column to No if they did not make it
            df["Tourney"][ind] = "No"
    
    # split data between features and values
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # use train_test_split to split the data (80% training data, 20% testing data)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, train_size = 0.8, random_state=0, shuffle = True)
    
    return X_train, X_test, y_train, y_test


def confusion_matrix(X_train, X_test, y_train, y_test):
    """
    
    Use logistic regression classifier to predict whether a team entered the NCAA
    tournament. Use a confusion matrix and heatmap to display the accuracy of the 
    classifier.
    
    Parameters
    ----------
    X_train : dataframe
        The values of the dataframe
    X_test : dataframe
        The split testing data
    y_train : series
        The feature which corresponds to the input values in X_train
    y_test : series
        The feature which corresponds to the input values from X_test

    Returns
    -------
    None.

    """
    # set classifier to logistic regression
    clf = LogisticRegression(class_weight = "balanced")
    clf.fit(X_train, y_train)
    
    # get predictions, and compare to y_test in classfification_report
    preds = clf.predict(X_test)
    report = metrics.classification_report(y_test, preds, digits = 2)
    print(report)
    
    # run confusion matrix to display accuracy
    matrix = metrics.confusion_matrix(y_test, preds)
    
    # display confusion matrix
    conf_df = pd.DataFrame(matrix)
    sns.heatmap(conf_df, annot=True)
    plt.xlabel("Guesses")
    plt.ylabel("True Labels")


def Main():
    
    print("Welcome to our final project!\n")
    
    # initialize the data into two data frames
    bas_df = read_csv(BASIC)
    adv_df = read_csv(ADVANCED)
    
    # merge the data frames into one and add tournament column
    all_df = merge_and_clean(bas_df, adv_df)
    
    # before performing logistic regression or plots, drop any columns which contains strings
    all_df = all_df.drop(["School"], axis = 1)
    
    # plot joinplots for stats we were interested in
    plot_joinplots(all_df)
    
    # plots a boxplot for each stat and making the NCAA tournament
    for col in all_df:
        boxplot(all_df, col)
    
    # split the data
    X_train, X_test, y_train, y_test = split_data(all_df)
    
    # train a classifier, show heat map displaying accuracy of results
    confusion_matrix(X_train, X_test, y_train, y_test)
        
    
if __name__ == "__main__":
    
    Main()
