#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Machine Learning Lab</h1>
# <h3 align="center">Lab 02</h3>
# <h3 align="center">Exercise 1 - Part A</h3>

# #### Importing Packages

# In[1]:


import pandas as pd      #Importing Pandas
import numpy as np       #Importing Numpy


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
merged_df = pd.merge(train_df, store_df, how='inner', on='Store')
merged_df.head()


# #### Cleaning and Preparing Dataframe for Analysis

# In[5]:


#Getting an overview which columns has null values
merged_df.isnull().sum()


# In[6]:


#For NA values in column Competition Distance, we are filling it with the median value of the whole column
merged_df.CompetitionDistance.fillna(merged_df.CompetitionDistance.median(), inplace = True)
#For NA in any other column, replacing NA with 0
merged_df.fillna(0, inplace = True)


# #### Find the store that has the maximum sale recorded. Print the store id, date and the sales on that day

# In[7]:


#Filtering rows on the basis of maximum Sales value
store_max_sale = merged_df[merged_df.Sales == merged_df.Sales.max()]
print('Store with Id: {} has the maximum sales on {} with value: {}\n'.format(
    store_max_sale.index[0],store_max_sale.Date.values[0],store_max_sale.Sales.values[0]))
store_max_sale


# #### Find the store(s) that has/ve the least possible and maximum possible competition distance(s).

# In[8]:


#Sorting the dataframe in descending on Competition Distance to get the Maximum and Minimum values
competition_distance = merged_df.sort_values('CompetitionDistance',ascending=False)
print('The Store with least Possible Competition Distance is :{} \n'.format(competition_distance.iloc[-1].name))
print(competition_distance.iloc[-1])
print('\nThe Store with Maximum Possible Competition Distance is : {} \n'.format(competition_distance.iloc[0].name))
print(competition_distance.iloc[0])


# #### What has been the maximum timeline a store has ran a "Promo" for? Which store was that, and what dates did the promotion covered?

# In[9]:


#Sorting the Merged Dataframe on Store and then Date columns and then grouping it on the basis of Store Id
store_promo_df = merged_df.sort_values(by=['Store','Date']).groupby(by='Store')
store_promo_dict = {}

#For each Store Id we compute the maximum number of times 1 appears consecutively 
#and then storing the maximum times into a dictionary with store Id as a key
for group, value in store_promo_df:
    max_promo = np.where(
        value["Promo"].eq(1),
        value.groupby(value.Promo.ne(value.Promo.shift()).cumsum()).cumcount() + 1,
        0,
    ).max()
    store_promo_dict[group] = max_promo

#Converting the dictionary into the Dataframe with Store Id as Index
store_promo_df = pd.DataFrame.from_dict(store_promo_dict,orient='index',columns=['Max Timeline for Promo'])
store_promo_df.index.name = 'Store'
store_promo_df


# #### What is the difference in the mean of sales (across all stores) when offering a Promo and not?

# In[10]:


#Calculating mean of sales when Store was offering Promo
sales_with_promo = merged_df[merged_df.Promo == True].Sales.mean()
print('Mean of Sales when the Store was offering Promo is: {}'.format(sales_with_promo))

#Calculating mean of sales when Store was NOT offering Promo
sales_without_promo = merged_df[merged_df.Promo == False].Sales.mean()
print('Mean of Sales when the Store was NOT offering Promo is: {}'.format(sales_without_promo))

#Calculating Difference in mean of sales when Store was offering Promo and not
print('Difference in the mean of sales (across all stores) when offering a Promo and not is: {}'.format(sales_with_promo - sales_without_promo))


# #### Are there any anomalies in the data as in where the store was "Open" but had no sales recorded? or vice versa?

# In[11]:


#For anamolies I am checking two conditions:
#-If store is open and there is no state holiday but the total sales is 0
#-If store is closed or there is state holiday but the total sales is greater than 0
anamolies = merged_df[((merged_df.Open == True) & (merged_df.Sales <= 0) & (merged_df.StateHoliday == '0'))
                     | (((merged_df.Open == False) | (merged_df.StateHoliday != '0')) & (merged_df.Sales > 0))]
anamolies


# #### Which store type (’a’,’b’ etc.) has had the most sales?

# In[12]:


#To find Store type with most sales,
#-First we group our dataframe on the basis of store type
#-Then for each store type group, calculate its total sales
#-And lastly sort the resulting dataframe on the basis of sales
store_max_sales = merged_df.groupby(by='StoreType')
store_max_sale = store_max_sales.sum().sort_values(by='Sales',ascending=False)
store_max_sale


# In[13]:


print('The Store type with the maximumn Sales is: {} with Sales amount: {}'.format(store_max_sale.index[0],store_max_sale.Sales[0]))

