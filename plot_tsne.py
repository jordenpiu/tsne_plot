import matplotlib.pyplot as plt 
import numpy as np 
import torch 
from matplotlib import cm 
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA 

import h5py
filename = "feature.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    
    a_group_key = list(f.keys())[0] #all_feats
    
    
    # Get the data
    data = list(f[a_group_key])

    features = f.get('all_feats')[:]
    labels = f.get('all_labels')[:]
    classes = np.unique(labels)
    

    print('features shape: ', features.shape)
    print('labels shape: ', labels.shape)

    print('classes: ', classes)

#plot tsne
tsne = TSNE(2, verbose = 1)

tsne_proj = tsne.fit_transform(features)

cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize = (8,8))
num_categories = 10 
for lab in range(num_categories):
    indices = features == lab
    ax.scatter(tsne_proj[indices, 0],
               tsne_proj[indices, 1],
               #tsne_proj[indices, 2],
               c = np.array(cmap(lab)).reshape(1,4),
               label = lab,
               alpha = 0.3)
ax.legend(fontsize = 'large', markerscale = 2)
plt.show()
    