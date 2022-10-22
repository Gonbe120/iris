#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt


# In[11]:


import mpl_toolkits.mplot3d


# In[12]:


from sklearn.datasets import load_iris


# In[13]:


from sklearn.decomposition import PCA


# In[14]:


iris = load_iris()


# In[15]:


X = iris.data[:,:2]
y = iris.target


# In[16]:


x_min,x_max = X[:,0].min() - 0.5,X[:,0].max() + 0.5
y_min,y_max = X[:,1].min() - 0.5,X[:,1].max() + 0.5


# In[19]:


plt.figure(2,figsize = (8,6))
plt.clf()

plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Set1,edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())


# In[23]:


fig = plt.figure(1,figsize=(8,6))
ax = fig.add_subplot(111,projection = "3d",elev=-150,azim=110)

X_reduced = PCA(n_components = 3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    X_reduced[:,2],
    c=y,
    cmap = plt.cm.Set1,
    edgecolor ="k",
    s = 40,
)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
dirname = './notebook/data/output/'
filename = dirname + 'img.png'
plt.savefig(filename)
print('end')


# In[ ]:




