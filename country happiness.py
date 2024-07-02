#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: alexandra
"""

import pandas as pd
import os
import csv as csv
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # self explanatory
from sklearn.preprocessing import StandardScaler # for PCA
from sklearn.linear_model import LinearRegression # also self explanatory
from sklearn.model_selection import train_test_split # testing and training data
from sklearn.cluster import KMeans


# Opening file

wd = os.getcwd()
file_location = 'WorldHappiness_Corruption_2015_2020.csv'
file_path = os.path.join(wd, file_location)

header = []

with open(file_path) as file:    
    csvreader = csv.reader(file) # Reads file
    header = next(csvreader) # Reads header of file
    #print(header) # good

    rows = []
    
    for i in csvreader:
        rows.append(i) # reads in row from file
    

# Dataframe (organizes things into a spreadsheet)
df = pd.DataFrame(rows, columns=header)

#allcolumnsnumerical = ['Country', 'happiness_score', 'gdp_per_capita', 'family', 'health', 'freedom', 'generosity', 'government_trust', 'dystopia_residual', 'continent', 'Year', 'social_support', 'cpi_score']

numerical_col1 = ['gdp_per_capita', 'happiness_score', 'health','freedom'] # Parameters we are analayzing wrt to each other or to general happiness
# Looking to find what has the largest variance i think idk
# Get to 2D

scaler = StandardScaler()
df[numerical_col1] = scaler.fit_transform(df[numerical_col1]) # Calculates std mean, then scales data; transforms data

# Doing PCA
pca1 = PCA(n_components = 2) # Breaking down components (GDP, happiness, health, CPI into 2D)
pca2 = PCA(n_components = 4)

principal_components1 = pca1.fit_transform(df[numerical_col1])
principal_components2 = pca2.fit_transform(df[numerical_col1])


print("principal component 1", principal_components1)
print("principal component 2", principal_components2)
explained_variance1 = pca1.explained_variance_ratio_ # Shows amount of variance for each component
explained_variance2 = pca2.explained_variance_ratio_ # Shows amount of variance for each component


print("variance 1", explained_variance1)
print("variance 2", explained_variance2) 


# LINEAR REGRESSION MODELS

# When GDP influences happiness the most
x1 = df[['gdp_per_capita']] # Takes GDP situation
y1 = df[['happiness_score']] # Takes the happiness situation

# Training and testing the data (GDP and happiness)
# 80/20 split (20% of the data we are testing, 80% we are using)
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size = 0.2, train_size = None, random_state = None)

# Linear regression model for GDP and happiness, as well as the line for linear regression
lr_1 = LinearRegression(fit_intercept = True)
lr_1.fit(x1_train, y1_train) # Use training data next time
y1_prediction = lr_1.predict(x1_test) # Predicts happiness scores based on GDP by taking in testing data

# Getting linear regression line
slope1 = lr_1.coef_[0][0]
intercept1 = lr_1.intercept_[0]
equation1 = f'Happiness = {slope1:.2f} * GDP + {intercept1:.2f}' # wrt to GDP
print(equation1)


# Plotting linear regression model for GDP and happiness
plt.scatter(x1_test, y1_test, color = 'g', label = 'GDP vs Happiness')
plt.plot(x1_test, y1_prediction, color = 'black', linewidth = 1.1, label = 'Regression line')
plt.xlabel('GDP')
plt.ylabel('Happiness')
plt.title('Linear regression for GDP vs Happiness')
plt.legend(fontsize = 'small')
plt.show()

# When Health influences happiness the most
x2 = df[['health']] # Takes health situation
y2 = df[['happiness_score']] # Takes the happiness situation

# Training and testing the data (health and happiness)
# 80/20 split (20% of the data we are testing, 80% we are using)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size = 0.2, train_size = None, random_state = None)

# Linear regression model for health and happiness
lr_2 = LinearRegression(fit_intercept = True)
lr_2.fit(x2_train, y2_train)
y2_prediction = lr_2.predict(x2_test) # Predicts happiness scores based on health by taking in testing data

# Getting linear regression line
slope2 = lr_2.coef_[0][0]
intercept2 = lr_2.intercept_[0]
equation2 = f'Happiness = {slope2:.2f} * Health + {intercept2:.2f}' # wrt to health
print(equation2)

# Plotting linear regression model for health and happiness
plt.scatter(x2_test, y2_test, color = '#F5E244', label = 'Health vs Happiness')
plt.plot(x2_test, y2_prediction, color = 'black', linewidth = 1.1, label = 'Regression line')
plt.xlabel('Health')
plt.ylabel('Happiness')
plt.title('Linear regression for Health vs Happiness')
plt.legend(fontsize = 'small')
plt.show()

# When Freedom influences happiness the most
x3 = df[['freedom']] # Takes freedom situation
y3 = df[['happiness_score']] # Takes the happiness situation

# Training and testing the data (freedom and happiness)
# 80/20 split (20% of the data we are testing, 80% we are using)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size = 0.2, train_size = None, random_state = None)

# Linear regression model for freedom and happiness
lr_3 = LinearRegression(fit_intercept = True)
lr_3.fit(x3_train, y3_train)
y3_prediction = lr_3.predict(x3_test) # Predicts happiness scores based on freedom by taking in testing data

# Getting linear regression line
slope3 = lr_3.coef_[0][0]
intercept3 = lr_3.intercept_[0]
equation3 = f'Happiness = {slope3:.2f} * Freedom + {intercept3:.2f}' # wrt to freedom
print(equation3)

# Plotting linear regression model for freedom and happiness
plt.scatter(x3_test, y3_test, color = '#B9463A', label = 'Freedom vs Happiness')
plt.plot(x3_test, y3_prediction, color = 'black', linewidth = 1.1, label = 'Regression line')
plt.xlabel('Freedom')
plt.ylabel('Happiness')
plt.title('Linear regression for Freedom vs Happiness')
plt.legend(fontsize = 'small')
plt.show()

# Mean for all variables
mean_gdp = np.mean(x1)
mean_happiness = np.mean(y1)
mean_health = np.mean(x2)
mean_freedom = np.mean(x3)
print("mean gdp:", mean_gdp)
print("mean happiness:", mean_happiness)
print("mean health:", mean_health)
print("mean freedom:", mean_freedom)

# Correlation Matrix
correlation_matrix = df[['happiness_score', 'gdp_per_capita', 'health', 'freedom']].corr()
print(correlation_matrix)


# PCA was reduced to 2 variables
# Creating a scatter plot and biplot to show direction of the arrows, as well as creating the arrows
#plt.scatter(principal_components1[:, 0], principal_components1[:, 1], color = 'c') # 4D EV
#plt.scatter(principal_components2[:, 0], principal_components2[:, 1], color = '#F7BF6E') # 2D EV


# Biplot (PCA 1)
#plt.scatter(principal_components2[:, 0], principal_components2[:, 1], color = '#F7BF6E') # 2D EV
for i, component in enumerate(pca1.components_):
    plt.arrow(0,0, component[0], component[1], head_width = 0.2, head_length = 0.1, fc = '#57C5EC', ec = '#57C5EC')
    plt.text(component[0], component[1], f"Eigenvector {i+1}" , fontsize = 12)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Biplot for PCA (2D)')
plt.grid(True) 
plt.show()

# Biplot (PCA 2)
# just this by itself and no scatter plot will show a larger version of the eigenvectors
#plt.scatter(principal_components2[:, 0], principal_components2[:, 1], color = '#20D8D8') # 4D EV
# comment out the above to just analyze the EV
for i, component in enumerate(pca2.components_):
    plt.arrow(0,0, component[0], component[1], head_width = 0.2, head_length = 0.1, fc = '#57C5EC', ec = '#57C5EC')
    plt.text(component[0], component[1], f"Eigenvector {i+1}" , fontsize = 12)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Biplot for PCA (4D)')
plt.grid(True) 
plt.show()

# Doing some form of clustering on the data set
clustering_variables = df[['gdp_per_capita', 'health', 'freedom']] # Variables used for clustering against happiness
kmeans = KMeans(n_clusters = 4)
kmeans.fit(clustering_variables)
clustering = kmeans.labels_ # creates a kmeans model

# Analyzing clusters (i.e., defining if they were good clusters) (ignore because idk honestly)
inertia = kmeans.inertia_
print(f"inertia: {inertia}") # Lower inertia = better clusters


# Lines for graphing a line through cluster
#gdpline = np.linspace(clustering_variables['gdp_per_capita'].min(), clustering_variables['gdp_per_capita'].max)
#healthline = np.linspace(clustering_variables['health'].min(), clustering_variables['health'].max())
#freedomline = np.linspace(clustering_variables['freedom'].min(), clustering_variables['freedom'].max())

# Center of clusters
cluster_centers = kmeans.cluster_centers_
print("cluster centers:\n", cluster_centers) # shows 4 rows (meaning 4 clusters) of clusters, in order: gdp, health, freedom, these show the mean value for each cluster
cluster_center_gdp = [-0.30984239, 0.33838999, -1.4299746, 1.01813682]
cluster_center_health = [-0.07407678, 0.21806561, -1.49075819, 0.96264991]
cluster_center_freedom = [0.55821034, -0.98610716, -0.67037662, 0.8588492]
cluster_labels_gdp = ['yes', 'Cyes', 'C', 'hjafan']
cluster_labels_health = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
cluster_labels_freedom = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

clusteringcolors = ['red', 'orange', 'yellow', 'green']


# Plotting clustering
# GDP vs Health cluster
plt.scatter(clustering_variables['gdp_per_capita'], clustering_variables['health'], c = clustering, cmap = 'Paired', label = cluster_labels_gdp)
plt.xlabel('GDP')
plt.ylabel('Health')
plt.title('GDP vs Health Clusters')
plt.scatter(cluster_center_gdp, cluster_center_health, c='red', marker='*', s=200, label='Cluster Centers')

# Labeling cluster mean values with stars (keep this one)
for i in range(len(cluster_center_gdp)):
    plt.text(cluster_center_gdp[i], cluster_center_health[i], '', fontsize=10, ha='center')
plt.show()

# GDP vs Freedom Cluster
plt.scatter(clustering_variables['gdp_per_capita'], clustering_variables['freedom'], c = clustering, cmap = 'Set2', label = cluster_labels_health)
plt.xlabel('GDP')
plt.ylabel('Freedom')
plt.title('GDP vs Freedom Clusters')


plt.scatter(cluster_center_gdp, cluster_center_freedom, c='red', marker='*', s=200, label='Cluster Centers')

# Labeling cluster mean values with stars (keep this one)
for i in range(len(cluster_center_gdp)):
    plt.text(cluster_center_gdp[i], cluster_center_freedom[i], '', fontsize=10, ha='center')
plt.show()

# Health vs Freedom Cluster
plt.scatter(clustering_variables['health'], clustering_variables['freedom'], c = clustering, cmap = 'PRGn', label = cluster_labels_freedom)
plt.xlabel('Health')
plt.ylabel('Freedom')
plt.title('Health vs Freedom Clusters')

plt.scatter(cluster_center_gdp, cluster_center_health, c='red', marker='*', s=200, label='Cluster Centers')

# Labeling cluster mean values with stars (keep this one)
for i in range(len(cluster_center_gdp)):
    plt.text(cluster_center_health[i], cluster_center_freedom[i], '', fontsize=10, ha='center')
plt.show()

# Close together clusters imply that there is more similarity in the data


# What clusters imply:
"""
GDP vs Health: 
    Cluster 1 implies poor health and GDP conditions in the countries
    Cluster 2 implies countries that may not be as developed, but have moderate and stable economic and health conditons
    Cluster 3 implies possibly severely poorer health conditions and severely poorer GDP conditions in the country
    Cluster 4 implies more developed countries, with possibly better economic and health resources

GDP vs Freedom:
    Cluster 1 implies poor economic conditions but a moderate amount of freedom
    Cluster 2 implies moderate amount of economic conditions, could have possibly been gained from various resources, but has low levels of freedom, possibsly some sort of authoritarian regime
    Cluster 3 implies countries with poor economic resources and levels of freedom; possibly under some sort of authoritarian regime/experiencing political instability
    Cluster 4 implies countries with high levels of freedom and economic resources, indicating robust economies and well-established democracies
    
Health vs Freedom:
    Cluster 1 implies countries with moderate levels of freedom, but poor infrastructure to health resources
    Cluster 2 implies countries with moderate health access, but poor levels of freedom (more political restrictions)
    Cluster 3 implies countries with poor levels of freedom and health access
    Cluster 4 implies countries with high levels of freedom and high levels of health
    
    In general: can interpret this as countries with high health, high freedom --> happier, more developed countries
"""

file.close() # Closes file
    
