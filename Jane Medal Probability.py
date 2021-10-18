#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font",size=14)


# In[2]:


# Import data for all Olympic athletes
olympics=pd.read_csv(r'C:\Users\jenni\Desktop\CODING_CERT\MSDS692\athlete_events_clean.csv',header=0)
olympics.head()


# In[3]:


# Gold medal count
medal_ranks=olympics[olympics['Medal']=='Gold'].groupby(['NOC','Year','Sport','Event','Season'])
medal_ranks=medal_ranks.first()
medal_ranks=medal_ranks.reset_index()
medal_ranks['NOC'].value_counts()


# In[5]:


# Filter data down to just Figure Skaters - Jane's Sport
Figure_Skating=olympics[olympics['Sport']=='Figure Skating']
Figure_Skating


# In[6]:


# Create medal count for Figure Skating by Country in order to find most historically successful nation in the sport
medal_Figure_Skating = Figure_Skating[Figure_Skating['Medal'].notnull()].groupby(['Year', 'Sport', 'Event'])
medal_Figure_Skating = medal_Figure_Skating.first()
medal_Figure_Skating = medal_Figure_Skating.reset_index()
medal_Figure_Skating['NOC'].value_counts()


# In[7]:


# Filter out null values for Height and Weight
Figure_Skating=Figure_Skating[Figure_Skating['Height'].notnull()]
Figure_Skating=Figure_Skating[Figure_Skating['Weight'].notnull()]
Figure_Skating


# In[8]:


# Make medal data binary - 0 for no medal and 1 for any medal
Figure_Skating['Medal'] = Figure_Skating['Medal'].fillna(0)
Figure_Skating['Medal'] = Figure_Skating['Medal'].replace('Bronze', 1)
Figure_Skating['Medal'] = Figure_Skating['Medal'].replace('Silver', 1)
Figure_Skating['Medal'] = Figure_Skating['Medal'].replace('Gold', 1)


# In[9]:


# Make Sex data binary - 0 for male and 1 for female
Figure_Skating['Sex'] = Figure_Skating['Sex'].replace('F', 1)
Figure_Skating['Sex'] = Figure_Skating['Sex'].replace('M', 0)


# In[10]:


pip install statsmodels


# In[11]:


pip install patsy


# In[12]:


import statsmodels.api as sm
from patsy import dmatrices


# In[13]:


# Create linear predictor based on Age
y, X = dmatrices('Medal ~ Age', data=Figure_Skating, return_type='dataframe')

# Create a GLM with said predictor, modeling the response as a Binomial
mod = sm.GLM(y, X, family=sm.families.Binomial())
res = mod.fit()
print(res.summary())


# In[14]:


# Create a function returning the logistic link function g(eta) for our Age model
def logisticReg(beta0, Age):
    return np.exp(beta0 + Age*X.Age)/(1 + np.exp(beta0 + Age*X.Age))

# Calculate the fitted values
y_fit = logisticReg(-4.2537, 0.1146)

# Plot the fitted model (red) against the observed values from the data (blue)
plt.plot(X.Age, y.Medal,'bo', label = 'Observed')
plt.plot(X.Age, y_fit,'c+', label = 'fit')
plt.legend(loc='best')
plt.xlabel('Age')
plt.ylabel('Medal Probability')
plt.title('Logistic Model of Medal outcomes of Figure Skaters based on Age')
plt.show()


# In[15]:


# Create linear predictor based on Height
y, X = dmatrices('Medal ~ Height', data=Figure_Skating, return_type='dataframe')

# Create a GLM with said predictor, modeling the response as a Binomial
mod = sm.GLM(y, X, family=sm.families.Binomial())
res = mod.fit()
print(res.summary())


# In[16]:


# Create a function returning the logistic link function g(eta) for Height model
def logisticReg(beta0, Height):
    return np.exp(beta0 + Height*X.Height)/(1 + np.exp(beta0 + Height*X.Height))

# Calculate the fitted values
y_fit = logisticReg(-2.278, 0.0039)

# Plot the fitted model (red) against the observed values from the data (blue)
plt.plot(X.Height, y.Medal,'bo', label = 'Observed')
plt.plot(X.Height, y_fit,'c+', label = 'fit')
plt.legend(loc='best')
plt.xlabel('Height')
plt.ylabel('Medal Probability')
plt.title('Logistic Model of Medal outcomes of Figure Skating based on Height')
plt.show()


# In[17]:


# Create linear predictor based on Age, Height, Weight and Sex
y, X = dmatrices('Medal ~ Age + Height + Weight + Sex', data=Figure_Skating, return_type='dataframe')
mod = sm.GLM(y, X, family=sm.families.Binomial())
res = mod.fit()
print(res.summary())


# In[18]:


# Create a function returning the logistic link function g(eta) for Age, Height, Weight and Sex model
def logisticReg2(beta0, Age, Height, Weight, Sex):
    eta = beta0 + Age*X.Age + Height*X.Height + Weight*X.Weight + Sex*X.Sex
    return np.exp(eta)/(1 + np.exp(eta))

y_fit = logisticReg2(-2.8801, 0.1225, -0.0141, 0.0114, 0.2834)

plt.plot(X.Height, y.Medal,'bo', label = 'Observed')
plt.plot(X.Height, y_fit,'c+', label = 'fit')
plt.legend(loc='best')
plt.xlabel('Height')
plt.ylabel('Medal Probability')
plt.title('Logistic Model of Medal outcomes of Figure Skating based on Height, Weight, Age and Sex')
plt.show()


# In[19]:


# Create binomial dummy figures for each Country with respect to Figure Skating
Figure_Skating=pd.get_dummies(Figure_Skating,columns=['NOC'],drop_first=True)
Figure_Skating


# In[20]:


# Create linear predictor based on Height, Weight, Sex, and representing the USA (highest ranking figure skating team)
y, X = dmatrices('Medal ~ Height + Weight + Sex + NOC_USA',
                 data=Figure_Skating, return_type='dataframe')
mod = sm.GLM(y, X, family=sm.families.Binomial())
res = mod.fit()
print(res.summary())


# In[21]:


# Create a function returning the logistic link function g(eta) for Height, Weight, Sex, and representing the USA (highest ranking figure skating team)

def logisticReg3(beta0, Height, Weight, Sex, USA):
  eta = beta0 + Height*X.Height + Weight*X.Weight + Sex*X.Sex + USA*X.NOC_USA
  return np.exp(eta)/(1 + np.exp(eta))

y_fit = logisticReg3(-1.6434, -0.0097, 0.0242, 0.2952, 0.3993)

plt.plot(X.Height, y.Medal + np.random.uniform(0,0.1,len(y.Medal)),'bo', 
         label = 'Observed')
plt.plot(X.Height, y_fit,'c+', label = 'fit')
plt.legend(loc='best')
plt.xlabel('Height')
plt.ylabel('Medal Probability')
plt.title('Logistic Model of Medal outcomes of Figure Skaters based on Height, Weight, Sex, and representing the USA')
plt.show()


# In[ ]:




