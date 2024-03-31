#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("C:/Users/lavan/Downloads/titles.csv")
df.head()  ### Top 5 records


# In[3]:


df.tail()  ### bottom 5 records


# In[8]:


df.loc[0] ### indexing in pandas


# In[11]:


df.loc[[10,100,1000,10000],['title']]


# In[4]:


df.loc[[10,100,1000,10000],['year']]


# In[5]:


df.loc[0:15] ### slicing of data


# In[6]:


df.loc[0:15:2]


# In[7]:


df.iloc[0:15] ### loc &iloc are same iloc does not read the text format


# In[10]:


df.iloc[[0,10,100,1000],[1,0]]


# In[11]:


df.iloc[100:120] ## it includes 100 & excludes 120 in iloc


# In[12]:


### statical technique called as correlation
### co-means two, relation means - association b/w two variabless


# correlation

# In[13]:


tips = pd.read_csv("D:/python practice notes/Tips.csv")
tips.head()


# In[15]:


#### correlation works on numerical variables
tips[['total_bill ','tip','size']].corr()


# In[16]:


import seaborn as sns


# In[20]:


sns.heatmap(tips[['total_bill ','tip','size']].corr(),annot = True)


# In[21]:


sns.heatmap(tips[['total_bill ','tip','size']].corr(),annot = True,cmap = 'RdBu')


# In[22]:


#### splitting of datasets
#### Train and test


# In[23]:


tips


# In[24]:


tips.shape


# In[26]:


tips.columns


# In[27]:


### independent variables
x = tips[['total_bill ','size']]
x.head(2)


# In[28]:


### independent variables
y = tips['tip']
y.head(2)


# In[29]:


from sklearn.model_selection import train_test_split


# In[31]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.15,random_state=365)


# In[32]:


x_train.head(3)


# In[33]:


##### Training of the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[34]:


### Train the model
model_train = model.fit(x_train,y_train)
print(model_train)


# In[35]:


pred = model_train.predict(x_test)
pred


# In[36]:


y_test


# In[38]:


###rsquare -r2 score (coefficient of Determination)
### it shows the strength of the model(0 to 1)
from sklearn.metrics import r2_score


# In[39]:


r2_score(y_test,pred)  ### model is weak


# In[ ]:




