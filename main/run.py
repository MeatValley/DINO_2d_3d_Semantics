import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import myutils.point_cloud as pc
import myutils.parse as parse
from dataset.semantics_dataset_min import SemanticsDataset
import matplotlib.pyplot as plt
import cv2
from feature_extractor.DINO_utils import get_feature_dictonary, intra_distance
import open3d as o3d
from myutils.clustering import test_with_Kmean
# from features.get_feature_from_file import get_feature_from_file

def run_Ayoub_adapted(file, number_images = 0, save_pc=False, K=12):
    """Runs Ayoub code adapted for other dataset
    
    """
    print(f'[running "run": reading {number_images+1} images from dataset, and Kmeans with K in range [2,{K}] ...]')
    config = parse.get_cfg_node(file)

    dataset = SemanticsDataset(config.data.path, config.data.point_cloud_name)

    dictionary = {}
    ind = number_images

    print("[getting a sample: ]")
    for i, sample in enumerate(iter(dataset)): #enumerate is just for i to be a counter
        print('[we are in a sample...]')
        print('image: ', i , end="\r")
        
        for j in range(len(sample["correspondances"])):
            if i == ind: break
            dictionary, patches = get_feature_dictonary(dictionary, sample["correspondances"][j], sample["image_DINO_features"][j], dataset)  
            # print('[trying to plot...]')
            # plt.figure()
            # plt.subplot(1, 5, 1)
            # plt.imshow(sample["image"][j])
            # # plt.subplot(1, 5, 2)
            # # plt.imshow(cv2.dilate(sample["proj3dto2d"], np.ones((2, 2), 'uint8'), iterations=4))
            # # plt.imshow(sample["proj3dto2d_depth"][j])
           
            # plt.subplot(1, 5, 3)
            # # plt.imshow(np.where(sample["proj3dto2d"] != 0, sample["proj3dto2d"], sample["image"]))
            # plt.imshow(sample["image"][j]-sample["proj3dto2d"][j])
            # # plt.imshow(sample["image"]-sample["proj3dto2d"])
            # plt.subplot(1, 5, 4)
            # plt.imshow(sample["proj3dto2d"][j])
            
                
            # # plt.subplot(1, 5, 5)
            # # plt.imshow(sample["depth_map"][j])
            # # plt.imshow(np.where(sample["proj3dto2d"] != 0, sample["proj3dto2d"], np.expand_dims(sample["depth_map"], axis=-1)))
            # plt.show()
        
        
        dataset.point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        #     # print("normals : ", np.asarray(dataset.point_cloud.normals))
        keys_sel = {}
        raw_features = []
        finale_features = []
        points = []
        colors = []
        normals = []


        for j, point in enumerate(dataset.point_cloud_points): # for each point
            if j in dictionary.keys():
        #         # print('[in the j dictionary...]') 
        #         print('[dico j]', dictionary[j] )

                mean = sum(dictionary[j])/len(dictionary[j]) #mean of features
                raw_features.append(mean)
        #         #for each point we can have more than 1 feature stacked in dictionary, so make a mean

                feat_temp = dictionary[j]
                feat_temp_dist=[np.linalg.norm(robust_feature - mean) for robust_feature in dictionary[j]]
        # #             #distance of each feature to the mean

                n_new_feat = max([int(0.6*len(feat_temp)), 1])
        #             # we catch just the ones near
                feat_temp_filtr = sorted(zip(feat_temp, feat_temp_dist), key= lambda x: x[1])[0:n_new_feat]
                feat_temp_filtr = [x[0] for x in feat_temp_filtr]                    # just add the ones near

                robust_mean = sum(feat_temp_filtr)/len(feat_temp_filtr)

        # #             #new mean (robust mean)
                keys_sel[len(points)]=j
                point = np.asarray(point)
                normal = np.asarray(dataset.point_cloud.normals[j])
                color = dataset.point_cloud_colors[j]/255
                        
        # #             # dino = mean
                dino = robust_mean
                points.append(point[None, :]) # [None, :] treats point as a single element
        # #             # x -> [x]
                colors.append(color[None, :])
                normals.append(normal[None, :])
        # #             #creates the finale feature

                feature  = dino[None, :]
        # #             # feature  = color[None, :]
                finale_features.append(feature)

                    
            

        # print('filane_features so far...', finale_features)
        finale_features = np.concatenate(finale_features, axis=0)
        points = np.concatenate(points, axis=0) #axis = 0 treats each independent
        colors = np.concatenate(colors, axis=0)

            # finale_features = (finale_features - finale_features.mean(axis=0))/finale_features.std(axis=0)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        #     # point_cloud.estimate_normals(
        # #     #             search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # #     # print("normals : ", np.asarray(point_cloud.normals))
            
        print('[vizualiation of pc...]')
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(width = 1300, height = 700)
        vis.add_geometry(point_cloud)
        vis.run()  # user picks points
        vis.destroy_window()

            # # test_with_Kmean(raw_features, points, config, range_for_K=4)
        #K mean is not good, re-evaluete the crops
        # test_with_Kmean(finale_features, points, config, range_for_K=K, save_pc= save_pc, index = number_images)
        test_with_Kmean(raw_features, points, config, range_for_K=4)
    