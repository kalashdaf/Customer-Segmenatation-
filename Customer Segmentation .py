#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


pip install matplotlib


# In[6]:


pip install seaborn


# In[7]:


import seaborn as sns


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


df=pd.read_csv("C:\\Users\\USER\\Downloads\\Mall_Customers.csv")
df.head()


# In[10]:


df.shape


# In[12]:


df.describe()


# In[13]:


df.dtypes


# In[14]:


df.isnull().sum()


# In[15]:


plt.figure(1,figsize=(15,6))
n=0
for x in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    n +=1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.distplot(df[x],bins=20)
    plt.title('Displot of{}'.format(x))
plt.show()


# In[ ]:





# In[16]:


plt.figure(figsize=(15,5))
sns.countplot(y='Gender',data=df)
plt.show()


# In[17]:


plt.figure(1,figsize=(15,7))
n=0
for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    n +=1
    plt.subplot(1,3,n)
    sns.set(style='whitegrid')
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.violinplot(x=cols,y ='Gender',data=df)
    plt.title('Violin Plot')
plt.show()


# In[18]:


age_16_25 = df.Age[(df.Age >=16) & (df.Age <=25)]
age_26_35 = df.Age[(df.Age >=26) & (df.Age <=35)]
age_36_45 = df.Age[(df.Age >=36) & (df.Age <=45)]
age_46_55 = df.Age[(df.Age >=46) & (df.Age <=55)]
age_55above = df.Age[(df.Age >=56)]
                     
agex =["16-25","26-35","36-45","46-55","55+"]
agey =[len(age_16_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55above.values)]
                     
plt.figure(figsize=(15,6))
sns.barplot(x=agex,y=agey, palette='mako')           
plt.title('No. of customer and Age')
plt.xlabel("Age")
plt.ylabel('No. of customers')
plt.show()


# In[19]:


sns.relplot(x="Annual Income (k$)", y="Spending Score (1-100)",data=df)


# In[14]:


pip install scikit-learn


# In[19]:


X1= df.loc[:,["Age","Spending Score (1-100)"]].values 

from sklearn.cluster import KMeans
wcss = []
for k in range (1,11):
    kmeans = KMeans (n_clusters=k,init="k-means++")
    kmeans.fit (X1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth =2, color="blue",marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[20]:


kmeans = KMeans (n_clusters=4)

label = kmeans.fit_predict(X1)
print(label)


# In[21]:


print(kmeans.cluster_centers_)


# In[22]:


plt.scatter(X1[:,0], X1[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color ='black')
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[23]:


X2= df.loc[:,["Annual Income (k$)","Spending Score (1-100)"]].values 

from sklearn.cluster import KMeans
wcss = []
for k in range (1,11):
    kmeans = KMeans (n_clusters=k,init="k-means++")
    kmeans.fit (X1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth =2, color="blue",marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[31]:


kmeans = KMeans (n_clusters=5)

label = kmeans.fit_predict(X2)
print(label)


# In[32]:


print(kmeans.cluster_centers_)


# In[33]:


plt.scatter(X1[:,0], X1[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color ='black')
plt.title('Clusters of Customers')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.show()


# In[48]:


X3=df.iloc[:,1:]

wcss = []
for k in range (1,11):
    kmeans = KMeans (n_clusters=k,init="k-means++")
    kmeans.fit (X1)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss, linewidth =2, color="blue",marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




