#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Machine Learning Lab</h1>
# <h3 align='center'>Lab 03</h3>

# ## Exercise 2: Linear Regression with Gradient Descent

# ### Part A: (Datasets)

# #### Importing Packages

# In[1]:


import pandas as pd                  #Importing Pandas
import numpy as np                   #Importing Numpy
import matplotlib.pyplot as plt      #Importing Matplotlib


# #### Reading Wine Quality Dataset

# In[2]:


wine_quality = pd.read_csv('winequality-red.csv',delimiter=';')
wine_quality.head()


# #### Reading Airfare and Demand Dataset

# In[3]:


airfare_demand = pd.read_fwf('airq402.data',header=None)
airfare_demand.columns = ['City1','City2','Average Fare1','Distance','Average weekly passengers',
                          'market leading airline','market share','Average fare2','Low price airline',
                          'market share','price']
airfare_demand.head()


# #### Reading Parkison Dataset

# In[4]:


parkison = pd.read_csv('parkinsons_updrs.data')
parkison.head()


# #### Convert any non-numeric values to numeric values. For example you can replace a country name with an integer value or more appropriately use hot-one encoding. [Hint: use pandas.get_dummies].Please explain your solution.

# In[5]:


wine_quality.info()

We can see from the Wine Quality Dataset info that No Column is Non-Numeric, so we skip Hot One Encoding here
# In[6]:


airfare_demand.info()

Here we see that the Columns : {0,1,5,8} are Non-Numeric Columns, so converting these columns into suitable Hot one Encoding
# In[7]:


#Converting Non-Numeric Columns to to numeric Columns using Hot one encoding scheme
airfare_demand_encoded = pd.get_dummies(airfare_demand,columns=['City1','City2','market leading airline','Low price airline'])
airfare_demand_encoded.head()


# In[8]:


parkison.info()

We can see from the Parkisons Dataset info that No Column is Non-Numeric, so we skip Hot One Encoding here
# #### If required drop out the rows with missing values or NA. In next lectures we will handle sparse data, which will allow us to use records with missing values.

# In[9]:


wine_quality.isnull().sum()

There are No Empty and NA values in Wine Quality Dataset
# In[10]:


airfare_demand_encoded.isnull().sum()

There are also No Empty and NA values in Airfare Demand Dataset
# In[11]:


parkison.isnull().sum()

There are also No Empty and NA values in Parkisons Dataset
# #### Split the dataset into 80% Train set and 20% Test set.

# ##### Function to Split any Dataframe into Training Set and Test Set

# In[12]:


def split_dataset(dataset,label):
    #Creating X matrix by removing the label/Target column
    X = dataset.drop(label,axis=1)
    #Creating Y vector which include only the Label/Target Column
    Y = dataset[label].to_numpy()

    #Adding a Bias Column in the X Matrix
    X = np.append(np.ones(shape=(len(X),1)),X,axis=1)
    
    #Calculating number of rows to be copied in the Training Set according to 80:20 ratio
    total_training_rows = int(len(X)*0.8)
    
    #Splitting the Dataset into Training set and Test Set based on calculated rows
    X_train , Y_train = X[:total_training_rows,:] , Y[:total_training_rows].reshape(-1,1)
    X_test , Y_test = X[total_training_rows:,:] , Y[total_training_rows:].reshape(-1,1)
    
    return X_train,X_test,Y_train,Y_test


# ##### Splitting Wine Quality Dataset

# In[13]:


wine_Xtrain , wine_Xtest , wine_Ytrain , wine_Ytest = split_dataset(wine_quality,'quality')


# ##### Splitting Airfare Demand Dataset

# In[14]:


airfare_Xtrain , airfare_Xtest , airfare_Ytrain , airfare_Ytest = split_dataset(airfare_demand_encoded,'price')


# ##### Splitting Parkison Dataset

# In[15]:


parkison_Xtrain , parkison_Xtest , parkison_Ytrain , parkison_Ytest = split_dataset(parkison,'total_UPDRS')


# ### Part B: Linear Regression with Real-World Data

# In[16]:


#Initializing arrays to store loss difference and RMSE values in different number of Iterations
loss_difference_values = np.array([])
rmse_values = np.array([])


# #### Function to Calculate Loss Between Actual Y and Predicted Y

# In[17]:


def loss_function(X,Y,B):
    # L(B) = summation((y-ypred)^2)
    return np.sum(np.square(np.subtract(Y,X @ B)))


# #### Function to Calculate the Loss Difference based on Old Beta Values and New Beta Values

# In[18]:


def loss_difference(X,Y,B_old,B_new):
    # |L(B_old) - L(B_new)|
    return np.abs(loss_function(X,Y,B_old) - loss_function(X,Y,B_new))


# #### Function to Calculate the RMSE Loss between Actual Y and Predicted Y

# In[19]:


def rmse(X,Y,B):
    # RMSE = square_root(summation((y-ypred)^2)/N)
    return np.sqrt(np.sum(np.square(np.subtract(Y , X @ B)))/len(X))


# #### Function which returns the gradient of Loss function

# In[20]:


def dL(X,Y,B):
    # Derivation of Loss = -2 * X^t * (y - ypred)
    return -2 * (X.T @ (Y - X @ B))


# #### Function to Minimize Gradient Descent based on Total Iterations and Learning Rate

# In[21]:


def minimize_GD(X,Y,X_test,Y_test,imax,mu):
    #Using global arrays to store loss difference and rmse values for different Iterations
    global loss_difference_values , rmse_values
    
    #Emptying both Loss difference and RMSE arrays
    loss_difference_values , rmse_values = np.array([]) , np.array([])
    
    #Initializing beta with Zeros
    beta = np.zeros(shape=(len(X[0]),1))
    for i in range(imax):
        #Calculating new Beta values from previous beta values and gradient descent direction
        #Beta = Beta - learning_rate * gradient Descent based on Beta
        beta_ = beta - mu * dL(X,Y,beta)
        
        #Appending Loss difference between between previous and new Beta
        loss_difference_values = np.append(loss_difference_values,loss_difference(X,Y,beta,beta_))
        
        #Appending RMSE loss between actual Y and Predicted Y
        rmse_values = np.append(rmse_values,rmse(X_test,Y_test,beta_))
        
        #Copying new Beta value to old Beta value for Further Calculation
        beta = np.copy(beta_)
    return beta_


# #### Function to Plot Loss Difference and RMSE Loss

# In[22]:


def plot_loss(xvalues , loss_difference , rmse ,title_graph1 ,title_graph2 ,xlabel_graph1 ,xlabel_graph2 ,ylabel_graph1,ylabel_graph2):
    #Plotting Loss Difference Graph
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(121)
    ax.plot(xvalues,loss_difference)
    ax.set_title(title_graph1)
    ax.set_xlabel(xlabel_graph1)
    ax.set_ylabel(ylabel_graph1)
    
    #Plotting RMSE Loss
    ax1 = fig.add_subplot(122)
    ax1.plot(xvalues,rmse)
    ax1.set_title(title_graph2)
    ax1.set_xlabel(xlabel_graph2)
    ax1.set_ylabel(ylabel_graph2)


# #### Applying Different Iteration and Different mu values for the Optimum values

# In[23]:


imax_d = [100,500,1000]
mu_wine = [0.0000001,0.00000001,0.000000001]
mu_parkison = [0.00000001,0.000000001,0.0000000001]
mu_airfare = [0.000000000001,0.0000000000001,0.00000000000001]


# #### Applying Different Iterations on Wine Quality Dataset with mu=0.0000001

# In[24]:


for i in imax_d:
    #Calculating new beta using Gradient Descent using mu = 0.0000001
    beta = minimize_GD(wine_Xtrain,wine_Ytrain,wine_Xtest,wine_Ytest,i,0.0000001)
    
    #Plotting the Loss and RMSE with different Iterations
    plot_loss([j for j in range(i)],loss_difference_values,rmse_values,
              'Wine Quality Dataset | Graph of |f(xi−1)−f(xi)| with Respect to {} Iterations\n'.format(i),
              'Wine Quality Dataset | Graph of RMSE with Respect to {} Iterations\n'.format(i),
              'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# #### Applying Different Iterations on Parkison Dataset with mu=0.00000001

# In[25]:


for i in imax_d:
    #Calculating new beta using Gradient Descent using mu = 0.00000001
    beta = minimize_GD(parkison_Xtrain,parkison_Ytrain,parkison_Xtest,parkison_Ytest,i,0.00000001)
    
    #Plotting the Loss and RMSE with different Iterations
    plot_loss([j for j in range(i)],loss_difference_values,rmse_values,
              'Parkison Dataset | Graph of |f(xi−1)−f(xi)| with Respect to {} Iterations\n'.format(i),
              'Parkison Dataset | Graph of RMSE with Respect to {} Iterations\n'.format(i),
              'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# #### Applying Different Iterations on Airfare Demand Dataset with mu=0.000000000001

# In[26]:


for i in imax_d:
    #Calculating new beta using Gradient Descent using mu = 0.000000000001
    beta = minimize_GD(airfare_Xtrain,airfare_Ytrain,airfare_Xtest,airfare_Ytest,i,0.000000000001)
    
    #Plotting the Loss and RMSE with different Iterations
    plot_loss([j for j in range(i)],loss_difference_values,rmse_values,
              'Airfare Demand Dataset | Graph of |f(xi−1)−f(xi)| with Respect to {} Iterations\n'.format(i),
              'Airfare Demand Dataset | Graph of RMSE with Respect to {} Iterations\n'.format(i),
              'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')

From the above plots and visualization we can see that the minimum loss occurs after 1000 iterations.So selecting imax = 1000 for further experimentation
# #### Applying Different mu values on Wine Quality Dataset with imax = 1000

# In[27]:


for mu in mu_wine:
    #Calculating new beta using Gradient Descent using imax = 1000
    beta = minimize_GD(wine_Xtrain,wine_Ytrain,wine_Xtest,wine_Ytest,1000,mu)
    
    #Plotting the Loss and RMSE with different mu values
    plot_loss([i for i in range(1000)],loss_difference_values,rmse_values,
              'Wine Quality Dataset | Graph of |f(xi−1)−f(xi)| with with mu: {}\n'.format(mu),
              'Wine Quality Dataset | Graph of RMSE with mu: {}\n'.format(mu),
              'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# #### Applying Different mu values on Parkison Dataset with imax = 1000

# In[28]:


for mu in mu_parkison:
    #Calculating new beta using Gradient Descent using imax = 1000
    beta = minimize_GD(parkison_Xtrain,parkison_Ytrain,parkison_Xtest,parkison_Ytest,1000,mu)
    
    #Plotting the Loss and RMSE with different mu values
    plot_loss([i for i in range(1000)],loss_difference_values,rmse_values,
              'Parkison Dataset | Graph of |f(xi−1)−f(xi)| with mu: {}\n'.format(mu),
              'Parkison Dataset | Graph of RMSE with mu: {}\n'.format(mu),
              'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# #### Applying Different mu values on Airfare Demand Dataset with imax = 1000

# In[29]:


for mu in mu_airfare:
    #Calculating new beta using Gradient Descent using imax = 1000
    beta = minimize_GD(airfare_Xtrain,airfare_Ytrain,airfare_Xtest,airfare_Ytest,1000,mu)
    
    #Plotting the Loss and RMSE with different mu values
    plot_loss([i for i in range(1000)],loss_difference_values,rmse_values,
              'Airfare Demand Dataset | Graph of |f(xi−1)−f(xi)| with mu: {}\n'.format(mu),
              'Airfare Demand Dataset | Graph of RMSE with mu: {}\n'.format(mu),
              'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# ## Exercise 3: Steplength Control for Gradient Descent

# #### Function to Plot Loss Difference and RMSE

# In[30]:


def plot_stepsize_iteration(xvalues1, xvalues2 , loss_difference , rmse ,title_graph1 ,title_graph2 ,
                            xlabel_graph1 ,xlabel_graph2 ,ylabel_graph1,ylabel_graph2):
    #Plotting Total Iteration vs Loss Difference Graph
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(121)
    ax.plot(xvalues1,loss_difference)
    ax.set_title(title_graph1)
    ax.set_xlabel(xlabel_graph1)
    ax.set_ylabel(ylabel_graph1)
    
    #Plotting different MU values vs RMSE Loss
    ax1 = fig.add_subplot(122)
    ax1.plot(xvalues2,rmse)
    ax1.set_title(title_graph2)
    ax1.set_xlabel(xlabel_graph2)
    ax1.set_ylabel(ylabel_graph2)


# #### Function to calculate optimum learning rate using Backtracking Algorithm

# In[31]:


def stepsize_backtracking(X,Y,B):
    mu = 1        #Starting Learning rate value
    alpha = 0.1   #Value for Alpha
    beta = 0.5    #Value for Beta
    
    #Iterate until the following condition is meet:
    #f (x) − f(x − µ∇f (x)) < αµ∇f (x)T ∇f (x)
    while((loss_function(X,Y,B) - loss_function(X,Y,B+(mu*-1*dL(X,Y,B))) < (alpha * mu * (dL(X,Y,B).T @ dL(X,Y,B))))):
        mu = mu * beta
    return mu


# #### Function to Minimize Gradient Descent using Backtracking algorithm

# In[32]:


def minimize_GD_backtracking(X,Y,X_test,Y_test,imax):
    #Creating and Initializing arrays to store loss difference and RMSE
    global loss_difference_values , rmse_values
    loss_difference_values , rmse_values = np.array([]) , np.array([])
    
    #Initializing beta with 0s
    beta = np.zeros(shape=(len(X[0]),1))
    for i in range(imax):
        #Calculating new Beta using Gradient Descent and backtracking algorithm
        beta_ = beta - stepsize_backtracking(X,Y,beta) * dL(X,Y,beta)
        
        #Calculating Loss difference between old Beta and New Beta
        loss_difference_values = np.append(loss_difference_values,loss_difference(X,Y,beta,beta_))
        
        #Calculating RMSE loss based on new Beta
        rmse_values = np.append(rmse_values,rmse(X_test,Y_test,beta_))
        
        #Copying New beta to Old Beta for Next Iteration
        beta = np.copy(beta_)
    return beta_


# In[33]:


#Calculating new beta using Gradient Descent using imax = 1000
beta = minimize_GD_backtracking(wine_Xtrain,wine_Ytrain,wine_Xtest,wine_Ytest,1000)
    
#Plotting the Loss and RMSE with different mu values
plot_stepsize_iteration([i for i in range(1000)],[j for j in range(1000)],loss_difference_values,rmse_values,
          'Wine Quality Dataset | Graph of |f(xi−1)−f(xi)| with 1000 Iterations\n'.format(i),
          'Wine Quality Dataset | Graph of RMSE with 1000 Iterations\n'.format(i),
          'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# In[34]:


#Calculating new beta using Gradient Descent using imax = 1000
beta = minimize_GD_backtracking(parkison_Xtrain,parkison_Ytrain,parkison_Xtest,parkison_Ytest,1000)
    
#Plotting the Loss and RMSE with different mu values
plot_stepsize_iteration([i for i in range(1000)],[j for j in range(1000)],loss_difference_values,rmse_values,
          'Parkison Dataset | Graph of |f(xi−1)−f(xi)| with 1000 Iterations\n'.format(i),
          'Parkison Dataset | Graph of RMSE with 1000 Iterations\n'.format(i),
          'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# In[35]:


#Calculating new beta using Gradient Descent using imax = 1000
beta = minimize_GD_backtracking(airfare_Xtrain,airfare_Ytrain,airfare_Xtest,airfare_Ytest,1000)
    
#Plotting the Loss and RMSE with different mu values
plot_stepsize_iteration([i for i in range(1000)],[j for j in range(1000)],loss_difference_values,rmse_values,
          'Airfare Demand Dataset | Graph of |f(xi−1)−f(xi)| with Respect to different Iterations\n'.format(i),
          'Airfare Demand Dataset | Graph of RMSE with Respect to different Iterations\n'.format(i),
          'Different Iterations','Different Iterations','|f(xi−1)−f(xi)|','RMSE')


# #### Function to calculate optimum learning rate using Bold Driver Algorithm

# In[36]:


def steplength_bolddriver(X,Y,B,alpha_old,alpha_plus,alpha_minus):
    #Increasing alpha value using alpha+
    alpha = alpha_old*alpha_plus
    
    #Iterating until following condition is meet:
    #f(x) − f(x + µd) ≤ 0
    while(loss_function(X,Y,B) - loss_function(X,Y,B - (alpha * dL(X,Y,B))) <= 0):
        #Slowly Increasing alpha using alpha-
        alpha = alpha * alpha_minus
    return alpha


# #### Function to Minimize Gradient Descent using Bold Driver algorithm

# In[37]:


def minimize_GD_bolddriver(X,Y,X_test,Y_test,imax):
    #Creating and Initializing arrays to store loss difference and RMSE
    global loss_difference_values , rmse_values
    loss_difference_values , rmse_values = np.array([]) , np.array([])
    
    #Initializing beta with 0s
    theta = np.zeros(shape=(len(X[0]),1))
    alpha = 1
    for i in range(imax):
        #Calculating alpha based bold driver algorithm
        alpha = steplength_bolddriver(X,Y,theta,alpha,1.1,0.5)
        
        #Calculating new Beta using Gradient Descent and backtracking algorithm
        theta_ = theta - alpha * dL(X,Y,theta)
        
        #Calculating Loss difference between old Beta and New Beta
        loss_difference_values = np.append(loss_difference_values,loss_difference(X,Y,theta,theta_))
        
        #Calculating RMSE loss based on new Beta
        rmse_values = np.append(rmse_values,rmse(X_test,Y_test,theta_))
        
        #Copying New beta to Old Beta for Next Iteration
        theta = np.copy(theta_)
    return theta_


# In[38]:


#Calculating new beta using Gradient Descent using imax = 1000
beta = minimize_GD_bolddriver(wine_Xtrain,wine_Ytrain,wine_Xtest,wine_Ytest,1000)
    
#Plotting the Loss and RMSE with different mu values
plot_stepsize_iteration([i for i in range(1000)],[j for j in range(1000)],loss_difference_values,rmse_values,
          'Wine Quality Dataset | Graph of |f(xi−1)−f(xi)| with 1000 Iterations\n'.format(i),
          'Wine Quality Dataset | Graph of RMSE with 1000 Iterations\n'.format(i),
          'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# In[39]:


#Calculating new beta using Gradient Descent using imax = 1000
beta = minimize_GD_bolddriver(parkison_Xtrain,parkison_Ytrain,parkison_Xtest,parkison_Ytest,1000)
    
#Plotting the Loss and RMSE with different mu values
plot_stepsize_iteration([i for i in range(1000)],[j for j in range(1000)],loss_difference_values,rmse_values,
          'Parkison Dataset | Graph of |f(xi−1)−f(xi)| with 1000 Iterations\n'.format(i),
          'Parkison Dataset | Graph of RMSE with 1000 Iterations\n'.format(i),
          'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')


# In[40]:


#Calculating new beta using Gradient Descent using imax = 1000
beta = minimize_GD_bolddriver(airfare_Xtrain,airfare_Ytrain,airfare_Xtest,airfare_Ytest,1000)
    
#Plotting the Loss and RMSE with different mu values
plot_stepsize_iteration([i for i in range(1000)],[j for j in range(1000)],loss_difference_values,rmse_values,
          'Airfare Demand Dataset | Graph of |f(xi−1)−f(xi)| with 1000 Iterations\n'.format(i),
          'Airfare Demand Dataset | Graph of RMSE with 1000 Iterations\n'.format(i),
          'Number of Iterations','Number of Iterations','|f(xi−1)−f(xi)|','RMSE')

Based on Backtracking and Bold Driver Algorithm performance, I think Bold driver did a bit better job as compared to Backtracting as it converges faster than the later But with a very smaller margin.The margin is so small that we can use any of these algorithm without any significant performance drop.