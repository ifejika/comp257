#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import numpy as np

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X_train, y_train = mnist.data[:60000], mnist.target[:60000]

# Create the instance of PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Output explained variance ratio of the 1st and 2nd components
explained_variance_ratio = pca.explained_variance_ratio_
print("1st component variance ratio:", explained_variance_ratio[0])
print("2nd component variance ratio:", explained_variance_ratio[1])


# In[2]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.astype(int), cmap='viridis', s=5, alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Projections of 1st and 2nd Principal Components onto 1D Hyperplane')
plt.colorbar()
plt.show()


# In[3]:


from sklearn.decomposition import IncrementalPCA

# Apply Incremental PCA
ipca = IncrementalPCA(n_components=154, batch_size=500)
X_ipca = ipca.fit_transform(X_train)

# Display variance ratio explained by 154 components
print(f"Explained variance ratio (154 components): {np.sum(ipca.explained_variance_ratio_)}")


# In[8]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.decomposition import IncrementalPCA

def plot_digits(data, n_rows=5, n_cols=5):
    plt.figure(figsize=(10, 10))
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(data[i].reshape(28, 28), cmap='binary')
        plt.axis('off')
    plt.show()

# Plot original and reduced dimensions
#plot_digits(X_train)  # Original
#plot_digits(X_ipca)   # Compressed


# In[9]:


# part 2 
from sklearn.datasets import make_swiss_roll

# Generate Swiss roll dataset
X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.1)

# Plot the generated Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=t, cmap='Spectral')
ax.set_title('Swiss Roll Dataset')
plt.show()


# In[10]:


from sklearn.decomposition import KernelPCA

kernels = ['linear', 'rbf', 'sigmoid']

for kernel in kernels:
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X_swiss)
    
    # Plot the 2D projection
    plt.figure(figsize=(8, 6))
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap='Spectral')
    plt.title(f'Kernel PCA with {kernel} kernel')
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.show()


# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# Use kPCA with Logistic Regression
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Define pipeline with kPCA and Logistic Regression
clf = make_pipeline(KernelPCA(), LogisticRegression())

# Define parameter grid
param_grid = {
    "kernelpca__kernel": ['rbf', 'linear', 'sigmoid'],
    "kernelpca__gamma": np.logspace(-2, 2, 5)
}

# Perform GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train[:1000], y_train[:1000])

# Print best parameters
print("Best parameters:", grid_search.best_params_)


# In[ ]:




