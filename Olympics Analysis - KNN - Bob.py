#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[4]:


# load dataset - Bob KNN from R
Bob=pd.read_csv('C:/Users/jenni/Desktop/CODING_CERT/MSDS692/Bob_KNN.csv')


# In[5]:


Bob.head(10)


# In[6]:


Bob.info()


# In[7]:


# Convert Weight, Height, and Age to numeric
Bob['Height']=(Bob['Height']).astype(int)


# In[8]:


Bob['Weight']=(Bob['Weight']).astype(int)


# In[9]:


Bob['Age']=(Bob['Age']).astype(int)


# In[10]:


Bob.info()


# In[11]:


Bob.head()


# In[12]:


Bob.info()


# In[13]:


Bob.describe()


# In[14]:


_=sns.heatmap(Bob.corr())


# In[15]:


# trim dataset
Bob.drop(['Sex','Season','Team'],axis=1,inplace=True)


# In[16]:


Bob.head(10)


# In[17]:


# Define the target factor for predictions - in this case, "Sport"
cols=Bob.columns
target_col='Sport'
feat_cols=[c for c in cols if c != target_col]

X=Bob[feat_cols].values
y=Bob[target_col].values


# In[18]:


# Create training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Define and fit the model
model=KNeighborsClassifier(n_neighbors=150)
model.fit(X_train,y_train)


# In[20]:


# gather the predictations that our model made for our test set
preds = model.predict(X_test)

# display the actuals and predictions for the test set
print('Actuals for test data set')
print(y_test)
print('Predictions for test data set')
print(preds)


# In[21]:


# Calculate the accuracy of the model
print(model.score(X_test, y_test))


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))


# In[23]:


neighbors = np.arange(1, 200)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[24]:


# Loop over K values
for i, k in enumerate(neighbors):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = model.score(X_train, y_train)
    test_accuracy[i] = model.score(X_test, y_test)


# In[25]:


# Generate plot for optimal k value using actuals for dataset
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[26]:


Bob.describe()


# In[27]:


Bob.head(10)


# In[28]:


# normalize the X_train and X_test only
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

X_tr_norm = min_max_scaler.fit_transform(X_train)
X_te_norm = min_max_scaler.fit_transform(X_test)


# In[29]:


# Normalize training data
new_Bob_tr = pd.DataFrame(X_tr_norm,columns=feat_cols)
new_Bob_tr['Sport'] = y_train
new_Bob_tr.head(10)


# In[30]:


# Normalize test data
new_Bob_te = pd.DataFrame(X_te_norm,columns=feat_cols)
new_Bob_te['Sport'] = y_test
new_Bob_te.head(10)


# In[49]:


# define and fit our model with k=1000
model_norm = KNeighborsClassifier(n_neighbors=1000, n_jobs=-1)
model_norm.fit(X_tr_norm, y_train)

# gather the predictations that our model made for our test set
preds_norm = model_norm.predict(X_te_norm)

# display the actuals and predictions for the test set
print('Actuals for test data set')
print(y_test)
print('Predictions for test data set')
print(preds_norm)


# In[50]:


# Calculate the accuracy of the normalized model
print(model_norm.score(X_te_norm, y_test))


# In[65]:


# Confusion matrix using normalized data
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, preds_norm))
print(classification_report(y_test, preds_norm))


# In[66]:


# Find optimal K value when using normalized data
n_neighbors = np.arange(1, 200)
train_accuracy_norm = np.empty(len(n_neighbors))
test_accuracy_norm = np.empty(len(n_neighbors))


# In[67]:


# Loop over K values
for i, k in enumerate(neighbors):
    model_norm = KNeighborsClassifier(n_neighbors=k)
    model_norm.fit(X_tr_norm, y_train)

    # Compute training and test data accuracy
    train_accuracy_norm[i] = model_norm.score(X_tr_norm, y_train)
    test_accuracy_norm[i] = model_norm.score(X_te_norm, y_test)


# In[36]:


# Generate plot to find optimal K value when using normalized data
plt.plot(neighbors, test_accuracy_norm, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy_norm, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[51]:


Orig_Model_Acc=(model.score(X_test, y_test))
Norm_Model_Acc=(model_norm.score(X_te_norm, y_test))

def maximum(a, b):
    a=Orig_Model_Acc
    b=Norm_Model_Acc
    if a >= b:
        return a
    else:
        return b

max=(maximum(Orig_Model_Acc,Norm_Model_Acc))
print('Accuracy of Original data is',Orig_Model_Acc,'and Accuracy of Normalized Model is',Norm_Model_Acc,". Therefore, we should use the model with a ",max," accuracy rate.")


# In[52]:


# take the first two features
X=Bob[feat_cols].values
y=Bob[target_col].values
h = .02  # step size in the mesh

# Calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Put the result into a color plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Height v. Weight")
plt.show()


# In[53]:


model=KNeighborsClassifier(n_neighbors=150,weights='distance')
model.fit(X_train,y_train)


# In[54]:


model.predict(X_train)


# In[55]:


import numpy as np
from sklearn import neighbors, datasets
from sklearn import preprocessing


# In[56]:


import six.moves
from six.moves import input as raw_input


# In[57]:


n_neighbors = 160

# prepare data
X=Bob[feat_cols].values
y=Bob[target_col].values
h = .02  # step size in the mesh

# we create an instance of Neighbours Classifier and fit the data.
model=KNeighborsClassifier(n_neighbors=110,weights='distance')
model.fit(X_train,y_train)


# make prediction
aa = raw_input('Enter Athlete Age: ')
ah = raw_input('Enter Athlete Height: ')
aw = raw_input('Enter Athlete Weight: ')
dataClass = model.predict([[aa,ah,aw]])
print('Prediction: '),

if dataClass == 'Track':
    print('Track')
elif dataClass == 'Swimming':
    print('Swimming')
elif dataClass == 'Wrestling':
    print('Wrestling')
elif dataClass == 'Judo':
    print('Judo')
elif dataClass == 'Gymnastics':
    print('Gymnastics')
elif dataClass == 'Table Tennis':
    print('Table Tennis')
else:
    print('Taekwondo')

