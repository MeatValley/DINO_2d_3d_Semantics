import numpy as np
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.image import load_image
from geometry.camera import Camera
import feature_extractor.DINO as DINO
import matplotlib.pyplot as plt


########################################################################################################################
#### 2d-3d semantics dataset
########################################################################################################################

class SemanticsDataset(Dataset):
    print('[creating a dataset...]')
    def __init__(self, root_dir, point_cloud_name, file_list = None, add_patch = None, scale = 0.25, features = "dino", device ="cpu"):
        self.scale = scale
        self.features = features
        self.add_patch = add_patch
        point_cloud_path = os.path.join(root_dir, point_cloud_name)
  
        self.data_transform = transforms.Compose( [ transforms.ToTensor(), transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5) ) ] )
        

        if file_list is not None:
            print("[file_list is not none]")
            self.split = file_list.split('/')[-1].split('.')[0]
            with open(file_list, "r") as f:
                self.color_split = f.readlines()  
        else:
            #get all images in the folder stream from "color" and with "depth"

            #list of color images 

            image_dir = os.path.join(root_dir, 'stream', 'color') #join the str in a path
            self.color_split = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                          if os.path.isfile(os.path.join(image_dir, f)) and 
                          (f.endswith('.png') or f.endswith('.jpg') 
                           or f.endswith('.jpeg'))]

            # #list of detph images
            depth_dir = os.path.join(root_dir, 'stream', 'depth')
            
            self.depth_split = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir)]
            
            # #list of poses matrix
            pose_dir = os.path.join(root_dir, 'stream', 'pose')            
            self.pose_var_split = [os.path.join(pose_dir, f) for f in os.listdir(pose_dir) 
                          if os.path.isfile(os.path.join(pose_dir, f)) and 
                          f.endswith('.json')]


            self.camera_k_matrix_split = []
            self.pose_split = []
            for json_file in self.pose_var_split:
                f = open(json_file)
                data = json.load(f)

                aux = data['camera_k_matrix']
                aux.append([0.,0.,0.])
                aux = np.array(aux)
                aux = np.concatenate((aux, [[0.], [0.], [0.], [1.]]), axis=1)
                self.camera_k_matrix_split.append(aux)


                temp = data['camera_rt_matrix']
                temp.append([0,0,0,1])
                self.pose_split.append(temp)

            self.root_dir = root_dir


            # #K for color img
            self.intrinsic = self.camera_k_matrix_split

            # #K for depth img
            self.intrinsic_depth = self.camera_k_matrix_split


            #translation correction between the 2 cameras
            
            #point_cloud in the data
            self.point_cloud = pc.load_point_cloud(point_cloud_path)
            self.point_cloud_points = np.asarray(self.point_cloud.points)
            self.point_cloud_colors = np.asarray(self.point_cloud.colors)


            self.device = torch.device(device)
            self.dino_model, self.patch_size = DINO.get_model('vits8')
            self.dino_model = self.dino_model.to(self.device)

            
            self.preprocess = transforms.Compose([
                                transforms.Resize(252, interpolation= transforms.InterpolationMode.BICUBIC),
                                transforms.FiveCrop(224),
                            ])
            self.preprocess2 = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])



########################################################################################################################

    @staticmethod
    def _get_pose(pose_file):
        return np.loadtxt(pose_file)

# ########################################################################################################################                            
    def __len__(self):
        """Dataset length"""
        return len(self.color_split)

########################################################################################################################
    @staticmethod
    def _get_intrinsics(intrinsic_file):
        """Get intrinsics from the calib_data dictionary."""
        return np.loadtxt(intrinsic_file)

########################################################################################################################

    def __getitem__(self, index):
        print(f'[getting a sample for the dataset with {index} index... ]')
        """Get dataset sample given an index."""
        image = load_image(self.color_split[index]) #1080x1080
        # plt.figure()
        # plt.imshow(image)
        # plt.show()


        #dimensions
        dims_image = tuple(np.asarray(image).shape[0:2][::-1])
        image = image.crop((20,20,1080-20,1080-20)) 

        original_image = image #1040x1040
        image_5crops = self.preprocess(image) #5x224x224
        image_res = image.resize((336, 336)) # 1080*0.4 = 432

        width, height = image_res.size

        pose = np.array(self.pose_split[index]) #get the pose of the current image
        # print('###############' ,pose)
        camera_intrinsic = np.array(self.camera_k_matrix_split[index])

        H, W = 224, 224 #size for dino
        new_width = W
        new_height = H

        crops = [(0, 0, new_width, new_height),
                 (width - new_width, 0, width, new_height),
                 (0, height - new_height, new_width, height),
                 (width - new_width , height - new_height, width, height),
                 ((width - new_width)/2, (height - new_height)/2, (width + new_width)/2, (height + new_height)/2)
                ]

        image_crop_tot = []
        projection3dto2d_tot = []
        pose_tot = []
        pc_im_correspondances_tot = []
        image_DINO_features_tot = []
        features_interpolation_tot = []

        for n, image_ts in enumerate(list(image_5crops)):
            print(f'[doing the process to the {n} crop..]')

            #getting features to the crops
            image_ts = self.preprocess2(image_ts)
            (left, top, right, bottom) = crops[n]
            image_crop = image_res.crop((left, top, right, bottom)) #image with 224, 224
            image_DINO_features_ts = DINO.get_DINO_features(self.dino_model.to('cpu'), image_ts)
            image_DINO_features = image_DINO_features_ts.detach().cpu().numpy().reshape(W//self.patch_size, H//self.patch_size, -1)
 
            #getting correspondences
            cam = Camera(K=camera_intrinsic, dimensions=dims_image, Tcw=np.linalg.inv(pose)).scaled(336./1080.).crop((left, top, right, bottom)) ####
            projection3dto2d, pc_im_correspondances = cam.project_on_image_min(
                self_pc = self.point_cloud_points, self_colors = self.point_cloud_colors)


            features_interpolation = None
            pose_tot.append(pose)
            image_crop_tot.append(np.asarray(image_crop)/255.0)
            image_DINO_features_tot.append(image_DINO_features) 
            features_interpolation_tot.append(features_interpolation) #use more than 1 feature, in this case not yet
            projection3dto2d_tot.append(projection3dto2d)
            pc_im_correspondances_tot.append(pc_im_correspondances)


            
        sample = {
            "original_image": original_image,
            'pose': pose_tot, #all TCW
            'image': image_crop_tot, #all img with 224x224
            "image_DINO_features": image_DINO_features_tot, #all (28,28,384) tuples of patch & feature
            "feature_interpolation": features_interpolation_tot,
            'proj3dto2d': projection3dto2d_tot,
            'correspondances' : pc_im_correspondances_tot
        }

        return sample
