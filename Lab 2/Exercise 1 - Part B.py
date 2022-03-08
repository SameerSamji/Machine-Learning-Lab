#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Machine Learning Lab</h1>
# <h3 align="center">Lab 02</h3>
# <h3 align="center">Exercise 1 - Part B</h3>

# #### Importing Packages

# In[1]:


import pandas as pd                 #Importing Pandas
import matplotlib.pyplot as plt     #Importing Matplotlib
import numpy as np                  #Importing Numpy


# #### Reading Training CSV data into the Dataframe

# In[2]:


train_df = pd.read_csv('train.csv',low_memory=False)
train_df['Date'] = pd.to_datetime(train_df['Date'])
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


# In[6]:


#Creating new columns Month,Year and Day from the Date column
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Year'] = merged_df['Date'].dt.year
merged_df['Day'] = merged_df['Date'].dt.day
merged_df


# #### On a monthly basis how do the mean of sales vary (across all stores)? plot these sale

# In[7]:


#To find mean sales across months we group by our dataframe on the basis of Months and calculate the mean sales
monthly_sales = merged_df.groupby(by='Month')['Sales'].mean()
print(monthly_sales)


# In[8]:


plt.plot(monthly_sales.index.values,monthly_sales)
plt.xlabel('Months')
plt.ylabel('Average Sales')
plt.title('Average Sales of Store across Months')
plt.show()


# #### On a daily basis how do the mean of sales vary (across all stores)? again, plot these sales.

# In[9]:


#To find mean sales across Days we group by our dataframe on the basis of Days and calculate the mean sales
daily_sales = merged_df.groupby(by='Day')['Sales'].mean()
print(daily_sales)


# In[10]:


plt.plot(daily_sales.index.values,daily_sales)
plt.xlabel('Day\'s')
plt.ylabel('Average Sales')
plt.title('Average Sales of Store across Day\'s')
plt.show()


# #### For the first store id, plot it’s cumulative sales for the first year.

# In[11]:


#Filtering our dataframe to include data for Store 1 only and then sort dataframe on Year column
merged_df[merged_df.index == 1].sort_values('Year').head()


# In[12]:


#Using the cumsum method to find the cumulative sales for store 1 in year 2013
cumulative_sum = merged_df[(merged_df.index == 1) & (merged_df.Year == 2013)].Sales.cumsum()


# In[13]:


plt.plot(np.arange(365),cumulative_sum)
plt.xlabel('Days')
plt.ylabel('Cumulative Sales')
plt.title('Cumulative Sales of Store 1 for the First Year')
plt.show()


# #### Plot and comment on the following relationships:

# ##### customers(x-axis) vs. sales(y-axis)

# In[14]:


plt.scatter(merged_df.Customers,merged_df.Sales)
plt.xlabel('Number of Customers')
plt.ylabel('Total Sales')
plt.title('Customers vs Sales')
plt.show()

The above graph shows the linear trend between Customers and Sales.If the number of customer increase, the sales also increases
# ##### competitiondistance(x-axis) vs. sales(y-axis)

# In[15]:


plt.scatter(merged_df.CompetitionDistance,merged_df.Sales)
plt.xlabel('Competition Distance')
plt.ylabel('Total Sales')
plt.title('Competition Distance vs Sales')
plt.show()

The above graph shows that when the competition distance is small, we have greater sales on average and vice versa
# #### Plot an array of Pearson correlations between all features. Remember to do the merge operation between the dataframes store and train.

# In[16]:


corr_matrix = merged_df.corr(method='pearson')
corr_matrix


# In[17]:


#Plotting our Pearson Correlation matrix as a heat map
import seaborn as sns      #Importing Seaborn
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr_matrix, annot = True,cmap= 'coolwarm',ax=ax)


# #### For the first 10 stores (id’ed) draw boxplots of their sales

# In[18]:


#Filtering our dataframe to include stores whose id is between 1 and 10
first_10_stores = merged_df[(merged_df.index >= 1) & (merged_df.index <= 10)]
#Selecting two columns from dataframe Store and Sales and sorting it with respect to Store Id
first_10_stores = first_10_stores[['Sales']].sort_values(by='Store')
first_10_stores


# In[19]:


#Plotting the Boxplot on Store and its Sales for first 10 Stores
first_10_stores.boxplot(by='Store',column =['Sales'],figsize=(10,10))


# #### From the above plot, which store has the highest median sales?
From the above graph, we can infer that Store 4 has the highest median Sales