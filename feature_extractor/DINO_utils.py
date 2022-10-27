import numpy as np
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_feature_dictonary(points_features, pc_im_correspondances, image_DINO_features, dataset, 
                     patch_size=8):
    print('[getting a DINO feature... ]')
    """get a mapping from point cloud pc to image im, and the DINO features from the dataset, receive a dictionary/points_feature that grows
    
    Parameters:
    points_features:
    pc_im_correspondences: for each good point in pc (key) its coordinates in the image
    
    output:
    points_features: for each point, its features (can be more than 1 since there is more photos)
    """
    patches = {}
    
    for key in pc_im_correspondances.keys(): #key is like a index for the points in pc
        #somewhere we map pc->im with index and this are the keys, and now we want the coordinates back
        im_coord_x, im_coord_y = pc_im_correspondances[key][0], pc_im_correspondances[key][1]
        # if key == 25576: print('##### correspondeces: ', im_coord_x, im_coord_y)

        patch_coord_x, patch_coord_y = im_coord_x//patch_size , im_coord_y//patch_size

        #some points may no appear in this image
        #the idea after is to do a mean
        if key not in points_features.keys():
            points_features[key] = []

        points_features[key].append(image_DINO_features[patch_coord_x, patch_coord_y]) #we have the 28 28, so here we put the 384 feature in  that point

        patches[key] = (patch_coord_x, patch_coord_y)

    return points_features, patches

def intra_distance(dico, perturb=False):

    distances = {}
    compteur=0
    
    length = len(list(dico.keys()))
    for k in dico.keys():
        nb_features = len(dico[k])
        
        # print("SHAPE : ", np.stack(dico[k]).shape)
        # tsne_plot(np.stack(dico[k]), k)
        
        if nb_features ==0:
            assert "error"
        if nb_features==1:
            distances[k] = 1.
            compteur = compteur+1
            continue
        if perturb:
            pert_index = random.choice(list(dico))
            nb_features_j = len(dico[pert_index])
            while nb_features_j == 1 or nb_features_j == 0:
                pert_index = random.choice(list(dico))
                nb_features_j = len(dico[pert_index])
            nb_features = min([nb_features, nb_features_j])
            # print(dico.keys)
            distance = [np.dot(dico[pert_index][j], dico[k][i]) / np.dot(dico[k][i], dico[k][i]) for i in range(nb_features) for j in range(nb_features) if i<j]
        else:
            distance = [np.dot(dico[k][j], dico[k][i]) / np.dot(dico[k][i], dico[k][i]) for i in range(nb_features) for j in range(nb_features) if i<j]
        distance = sum(distance)/len(distance)
        
        distances[k] = distance
    print("number of ones : ", compteur/len(dico.keys()))
    return distances


def tsne_plot(points_features, N_labels, dist=None, labels=None, perplexity=30, 
              n_components=2):
    """Plot the t-SNE embedding of the points_features.
        t-SNE is a representation of multidimensional vectors in 2D by
    
    """


    tsne = TSNE(n_components=n_components, perplexity=perplexity, verbose=1, n_iter=5000)
    tsne_results = tsne.fit_transform(points_features)
    print(tsne_results.shape)
    
    if labels is None:
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.colorbar(ticks=range(N_labels))
        plt.clim(-0.5, 9.5)
    elif dist is None:
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", N_labels))
        plt.colorbar(ticks=range(N_labels))
        # plt.clim(-0.5, 9.5)
    else:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        p = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], np.array(dist),  c=labels, cmap=plt.cm.get_cmap("jet", N_labels))
        fig.colorbar(p, ticks=range(N_labels))
    # plt.colorbar(ticks=range(10))
    
    plt.show()
    return tsne_results