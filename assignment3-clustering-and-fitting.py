# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:11:07 2023

@author: Kashir
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from err_ranges import err_ranges
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


def read_dataset(filename):
    """
    
    This function load dataset from same directory and transpose in and return
    two dataset one year as column and second country name
        
    """
    
    # Read csv and skip garbage values which is in first 4 rows
    df = pd.read_csv(filename, skiprows=4)
    
    # Drop unnamed column
    df = df.iloc[:,:-2]
    
    
    # Create a dataset with countries as columns
    by_countries = df.set_index(["Country Name", "Indicator Name"])
    by_countries.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)
    
    # Transpose the countries dataframe
    by_countries = by_countries.T
    
    # Return the years and countries dataframes
    return df, by_countries


def remove_extra_indicators(df, indicators):
    """
    This function is use to return only data data set with those indicators
    which we will used in our project
    
    """
    
    # This will return dataset with those indicator which we need to use
    filtered_dataset = df[df["Indicator Name"].isin(indicators)]
    
    
    return filtered_dataset


def bar_chart(df,labels):
    
    """
    This function is use to print the bar chart it takes dataset as input and
    labels
    
    """
    #countries list which we have to plot a bar chart
    countries_list = [
        "India",
        "China",
        "Italy",
        "France",
        "Israel",
        "United Kingdom",
        "Finland",
        "Japan",
        "United States"
        ]
    #reset index
    temp = df.reset_index()
    
    #set the dataset by only countries which we need to make analysis
    temp2 = temp[temp["Country Name"].isin(countries_list)]
    
    # Set the number of rows and columns for the subplot grid
    n_rows = int(len(labels)/2)
    n_cols = 2
    
    # Create the figure and subplots using plt.subplots()
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 30))
    
    # Flatten the array of subplots to simplify indexing
    axs = axs.flatten()
    
    # Loop over the labels and create a bar chart for each one
    for i, label in enumerate(labels):
        temp3 = temp2[["Country Name", label]].set_index("Country Name")
        temp3 = temp3.sort_values(by=label, ascending=True)
        ax = axs[i]
        temp3.plot(kind="bar", ax=ax)
        ax.set_title(label)
    
    # Use tight_layout to adjust the spacing between subplots
    plt.tight_layout()
    plt.show()

def clustering_function(dataset,number_of_cluster):
    
    """
    This function will accept dataset and then process normalization on it
    after normalization it perform clustering and after that make new column
    and store the value there in the dataset
    
    """
    #Normalize the dataset
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(dataset.values)
    
    # Choose the number of clusters
    n_clusters = number_of_cluster
    
    # Perform clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(normalized_data)
    
    # Add the cluster labels to the dataset
    dataset["Cluster"] = labels
    return dataset

def scatter_plot_for_clustered_data(df,x,y,xLabel,yLabel,title):
    
    # Create a scatter plot with the specified columns and labels
    plt.figure(figsize=(10, 5))
    plt.scatter(df[x],
                df[y],
                c=df['Cluster'])
    
    # Add axis labels and a title to the plot
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    
    # Show the plot
    plt.show()
    
    
if __name__=="__main__":
    #read dataset by calling the function
    df, countries = read_dataset("dataset.csv")
    
    #list of indicators which we need for our analysis
    indicators_list = [
    'Agricultural land (% of land area)',
    'Forest area (% of land area)',
    'CO2 emissions (kt)',
    'Methane emissions (kt of CO2 equivalent)',
    'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)',
    'Total greenhouse gas emissions (kt of CO2 equivalent)',
    'Renewable electricity output (% of total electricity output)',
    'Renewable energy consumption (% of total final energy consumption)',
    ]
    indicator_list_short = [
        'Agricultural land ',
        'Forest area ',
        'CO2 emissions (kt)',
        'Methane emissions',
        'Nitrous oxide emissions',
        'Total greenhouse gas emissions',
        'Renewable electricity output',
        'Renewable energy consumption',
        ]
    # getting only those data 
    filtered_dataset = remove_extra_indicators(df, indicators_list)
    
    #clean dataset by filling missing values
    filtered_dataset = filtered_dataset.fillna(method="ffill")\
        .fillna(method="bfill")
    
    # Pivot the dataset to get the values by indicator name with country name
    pivot_dataset = filtered_dataset.pivot_table(index='Country Name'
                                       , columns='Indicator Name'
                                       , values='2020')
    
    
    #bar_chart(pivot_dataset, indicators_list)
    
    #make dictionary to make label short
    label_dict = {
    'Agricultural land (% of land area)': 'Agri. land',
    'CO2 emissions (kt)': 'CO2 emissions',
    'Forest area (% of land area)': 'Forest area',
    'Methane emissions (kt of CO2 equivalent)': 'Methane emissions',
    'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)': 'Nitrous oxide emissions',
    'Total greenhouse gas emissions (kt of CO2 equivalent)': 'Total GHG emissions',
    'Renewable electricity output (% of total electricity output)':'Renewable electricity',
    'Renewable energy consumption (% of total final energy consumption)':'Renewable energy',
    }
    
    #rename labels to shortened the name to prevent label collapse
    pivot_dataset_scatter = pivot_dataset.rename(columns=label_dict)
    '''
    # scatter plot
    pd.plotting.scatter_matrix(pivot_dataset_scatter, figsize=(15.0, 15.0))
    plt.tight_layout()    # helps to avoid overlap of labels
    plt.xticks(rotation=90)
    plt.subplots_adjust(wspace=1.0, hspace=1.0)
    plt.show()
    
    # Correlation
    corr = pivot_dataset.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', xticklabels=indicator_list_short, yticklabels=indicator_list_short)
   
    plt.title('Correlation between environmental factors')
    plt.show()
    '''
    #clustering with 3
    clustered_dataset = clustering_function(pivot_dataset,3)
    
    #scatter plot with cluster  3
    scatter_plot_for_clustered_data(clustered_dataset
                                    ,'Forest area (% of land area)'
                                    ,'CO2 emissions (kt)'
                                    ,'Total greenhouse gas emissions (kt of CO2 equivalent)'
                                    ,'CO2 emissions (kt)'
                                    ,'CO2 emission vs Greenhouse emission by Cluster')
    #clustering with 4
    clustered_dataset_four = clustering_function(pivot_dataset,4)
    
    #scatter plot with cluster 4
    scatter_plot_for_clustered_data(clustered_dataset_four
                                    ,'Forest area (% of land area)'
                                    ,'CO2 emissions (kt)'
                                    ,'Total greenhouse gas emissions (kt of CO2 equivalent)'
                                    ,'CO2 emissions (kt)'
                                    ,'CO2 emission vs Greenhouse emission by Cluster')
    
    