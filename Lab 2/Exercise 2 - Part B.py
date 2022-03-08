#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Machine Learning Lab</h1>
# <h3 align="center">Lab 02</h3>
# <h3 align="center">Exercise 2 - Part B</h3>

# #### Importing Packages

# In[1]:


import pandas as pd      #Importing Pandas
import numpy as np       #Importing Numpy
import math              #Importing Math


# #### Reading Training CSV data into the Dataframe

# In[2]:


train_df = pd.read_csv('train.csv',low_memory=False)
#Setting Store column as my Index in Dataframe
train_df.set_index('Store',inplace=True)
train_df.head()


# #### Reading Store CSV data into the Dataframe

# In[3]:


store_df = pd.read_csv('store.csv')
#Setting Store column as my Index in Dataframe
store_df.set_index('Store',inplace=True)
store_df.head()


# #### Merging Training and Store Dataframes into a single Dataframe

# In[4]:


#Merging train and store dataframes on Store column
merged_df = pd.merge(train_df,store_df,how='inner',on='Store')
merged_df


# #### Cleaning and Preparing Dataframe for Analysis

# In[5]:


#For NA values in column Competition Distance, we are filling it with the median value of the whole column
merged_df.CompetitionDistance.fillna(merged_df.CompetitionDistance.median(), inplace = True)
#For NA in any other column, replacing NA with 0
merged_df.fillna(0, inplace = True)


# #### Find all the stores that have sales recorded for 942 days

# In[6]:


merged_df = merged_df[merged_df.groupby(by='Store').count() >= 942]
#Taking only Store and Sales Columns
merged_df = merged_df[['Sales']]
merged_df


# #### Create a data matrix of the shape (#_of_stores, 942) for the daily sales record of these stores.

# In[7]:


#First Grouping the dataframe by Store so to get (# of stores) rows
store_sales = merged_df.groupby(['Store'])

#For each store, converting its sales column to a list and storing it into a single cell
store_sales = store_sales['Sales'].agg(lambda x: list(x))

#Converting the list each cell into 942 different columns and each row now represent each store
store_sales = pd.DataFrame(store_sales.values.tolist(),columns=[x+1 for x in range(942)])
store_sales


# #### Use the first 800 stores in this data matrix for training and the rest for testing. Also split the sales data into 2 parts, the 1st part contains the information about the first 900 days of sales (these would be the features) and the 2nd contains the information about the last 42 days of sales (these would be the targets).

# In[8]:


#Using iloc to split dataset and then converting this to numpy array for further processing
X_train = store_sales.iloc[:800, :900].to_numpy()
X_test = store_sales.iloc[800:,:900].to_numpy()
Y_train = store_sales.iloc[:800,900:].to_numpy()
Y_test = store_sales.iloc[800:,900:].to_numpy()


# In[9]:


#Replacing every NaN value with 0 in training and test Dataset
X_train[np.isnan(X_train)] = 0
Y_train[np.isnan(Y_train)] = 0
X_test[np.isnan(X_test)] = 0
Y_test[np.isnan(Y_test)] = 0


# #### Iteratively build multiple linear regression models for column vectors of 𝑌𝑡𝑟𝑎𝑖𝑛. You are allowed to use the numpy routines for calculating inverses, transposing of matrices and matrix multiplication. You would need to create 42 models in this case (1 model for each day in the target sales matrix)

# In[10]:


#Calculating Matrix A which equals X^t.X
A = X_train.T @ X_train

#Creating a matrix which stores beta vector for next 42 days
all_beta = np.zeros(shape=(900,42))

#Iterating through the columns for Y train
for i in range(Y_train.shape[1]):
    #Initialize empty beta array
    beta = np.zeros(shape=(900,1))
    #Calculating beta vector using equation: (X^t.X)^-1.(X^t.Y)
    beta = np.array(np.dot(np.linalg.inv(A),np.dot(X_train.T,Y_train[:,i])))
    #Reshaping beta array to column vector
    beta = beta.reshape(-1,1)
    #Copying new beta values to beta matrix
    all_beta[:,i] = beta[:,0]


# #### Verify that you have learned 𝛽0:900 for each of the 42 models and use these learned parameters to make predictions for each day ahead. In total 42 days.

# In[11]:


#Calculating predicted y y hat using the X test and beta values
y_hat = X_train @ all_beta


# #### Calculate and print the daily RMSE and MAE for all 42 sales values using test split (𝑋𝑡𝑒𝑠𝑡 as input). Also calculate and print overall average RMSE and MAE. (i.e. just the mean RMSE of all 42 models).

# ##### Function to calculate RMSE between Actual Y and Predicted Y

# In[12]:


def calculate_rmse(actual,predicted):
    N = len(actual)
    summation = 0
    for i in range(len(actual)):
        summation += ((actual[i] - predicted[i]) ** 2)
    return math.sqrt(summation/N)


# ##### Function to calculate MAE between Actual Y and Predicted Y

# In[13]:


def calculate_mae(actual,predicted):
    N = len(actual)
    summation = 0
    for i in range(len(actual)):
        summation += abs(actual[i] - predicted[i])
    return summation/N


# In[14]:


#Initializing two empty array for storing RMSE and MAE
rmse = np.zeros(shape=(Y_train.shape[1],))
mae = np.zeros(shape=(Y_train.shape[1],))

#Iterating through Y values
for i in range(Y_train.shape[1]):
    rmse[i] = calculate_rmse(Y_train[:,i],y_hat[:,i])
    mae[i] = calculate_mae(Y_train[:,i],y_hat[:,i])

#Converting RMSE and MAE list to Dataframe
error_df = pd.DataFrame(np.hstack((rmse.reshape(-1,1),mae.reshape(-1,1))),columns=['RMSE','MAE'])
error_df


# #### Reason why or why not Linear Regression is a good choice for this task.
I think linear regression is not a good choice for this task, instead we can use time series forcast method Autoregressive model (AR) because it predict future behaviour based on past behaviour.The linear regression model uses the linear combination of predictors for predicting future values whereas AR model uses all the past values in time and then predict the future values.