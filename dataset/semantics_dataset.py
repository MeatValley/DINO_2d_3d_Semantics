import numpy as np
import os
import open3d as o3d

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
            # print(self.pose_split)

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


######################################################### supposing depth cam = color cam, no Tcolortodepth
            # #K for color img
            self.intrinsic = self.camera_k_matrix_split

            # #K for depth img
            self.intrinsic_depth = self.camera_k_matrix_split


            #translation correction between the 2 cameras

            # colortodepth_path = os.path.join(root_dir, os.path.basename(os.path.basename(root_dir))+'.txt')
            # # print("colortodepth_path: ", colortodepth_path)
            # # print(os.path.basename(root_dir))
            # # print(os.path.dirname(root_dir))
            # self.Tcolortodepth = self._get_colortodepth(colortodepth_path)
            
            #point_cloud in the data
            self.point_cloud = pc.load_point_cloud(point_cloud_path)
            # pcd = o3d.io.read_point_cloud(point_cloud_path, format="xyz")
            # print(pcd.points)
            # pc.show_point_cloud(pcd)
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
    @staticmethod
    def _get_colortodepth(colortodepth_file):
        print('[not supposed to use get_colortodepth...]')
        """Get color to depth from the calib_data dictionary."""
        # print("colortodepth_path: ", colortodepth_file)
        colortodepth = 'dont need for now'
        # with open(colortodepth_file, "r") as f:
        #     colortodepth = f.readlines()
        # colortodepth = colortodepth[2].split(" ")[2::]
        # colortodepth = np.array([[float(colortodepth[4*i+j]) for j in range(4)] for i in range(4)], dtype=np.float32)
        return colortodepth

########################################################################################################################

    def __getitem__(self, index):
        ###################################################################### not sure about the K and RT are what I tryed
        print(f'[getting a sample for the dataset with {index} index... ]')
        """Get dataset sample given an index."""
        image = load_image(self.color_split[index]) #1080x1080
        depth = load_image(self.depth_split[index])
        # depth.resize((640,480))

        img = depth
        image2 = np.zeros((1080, 1080, 1)) #224x224
        pixels = img.load() # create the pixel map

        # for i in range(img.size[0]):    # for every col:
        #     for j in range(img.size[1]):
        #         if pixels[i,j]>50000: print(pixels[i,j])





        #dimensions
        dims_image = tuple(np.asarray(image).shape[0:2][::-1])
        dims_depth = tuple(np.asarray(depth).shape[0:2][::-1]) # both 1080 x 1080
        image = image.crop((20,20,1080-20,1080-20)) 
        # print(image)

        original_image = image #1040x1040
        image_5crops = self.preprocess(image) #5x224x224
        image_res = image.resize((432, 432)) # 1080*0.4 = 432
        # image_res = image.resize((int(1292 * (252/968)), 252))

        #was 336x252 (not sure why)
        ######################################################### ayoub decision based on the original img, maybe we change
        width, height = image_res.size
        # plt.figure()
        # plt.imshow(image_res)
        # plt.show()

        depth = np.array(depth)/512 #in meter
        # depth[depth > 65534/512] = 0.000001

        # print('depth: ', depth, depth.shape)
        # print('depth map max: ', depth.max())
        # print('depth map min: ', depth.min())



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
        depth_tot = []
        projection3dto2d_depth_tot = []
        features_interpolation_tot = []

        for n, image_ts in enumerate(list(image_5crops)):
            print(f'[doing the process to the {n} crop..]')
            image_ts = self.preprocess2(image_ts)
            # print(image_ts.shape) #image_ts is eache crop with 224x224 for each rgb of each crop (preprocessate)
            # print("img_ts:", image_ts) 
            (left, top, right, bottom) = crops[n]
            image_crop = image_res.crop((left, top, right, bottom)) #image with 224, 224
            image_DINO_features_ts = DINO.get_DINO_features(self.dino_model.to('cpu'), image_ts)
            image_DINO_features = image_DINO_features_ts.detach().cpu().numpy().reshape(W//self.patch_size, H//self.patch_size, -1)
 
            # print('intrinsics', camera_intrinsic)
            cam = Camera(K=camera_intrinsic, dimensions=dims_image, Tcw=np.linalg.inv(pose)).scaled(0.4).crop((left, top, right, bottom)) ####
            # print('cam.K = ', cam.K)
            cam_depth = cam

            coord_depth = np.asarray([[i, j] for i in range(depth.shape[0]) for j in range(depth.shape[1])]) # 1080*1080*2 xy for 1080^2

            
            #     #the coord of each pixel
            # print('u v q entra ', coord_depth) # 0 0 -> 1080 1080
            # print(coord_depth_000000.size) # 1080*1080*2 xy for 1080^2
            value_depth = np.asarray([[depth[i,j]] for i in range(depth.shape[0]) for j in range(depth.shape[1])])
            # for d in value_depth:
            #     if d == 0: print('depth', d)
            # print('z  ', value_depth) #0.591 -> 1.104
            #     #its values
            # print('1          ', value_depth.size) #1080x1080 (the z for the coord)
            point_cloud_np = cam_depth.get_point_cloud(coord_depth, value_depth)  #(x,y,z, 1080*1080) point_xyz_world


            # point_cloudx = o3d.geometry.PointCloud()
            # point_cloudx.points = o3d.utility.Vector3dVector(point_cloud_np)
            # o3d.visualization.draw_geometries([point_cloudx], width = 1500, height = 800)



            # print('x y z', point_cloud_np, point_cloud_np.shape) #o Z q eu to recebendo aqui ta mto diferente do Z suposto
            ########################################################################################################################### ----------------
            projection3dto2d_depth = cam.project_on_image(X_pos = point_cloud_np, X_color = value_depth) #(224,224,1) for each pixel of the img that goes dino, its depth
            # print('u v q sai: ', projection3dto2d_depth[0], projection3dto2d_depth[1])
            # print('projection3dto2d_depth: ', projection3dto2d_depth, projection3dto2d_depth.shape)  #224x224x1 
                
            # print(projection3dto2d_depth.sum()) #empty -> no point cloud points represents a pixel, impossibel
            ###########################################################################################################################----------------
            # print('self points: ', self.point_cloud_points, self.point_cloud_points.shape)
            # self_proj_self_pc = cam.project(self.point_cloud_points)
            # print('self proj self pc: ', self_proj_self_pc)
            # print('self colors: ', self.point_cloud_colors)
            projection3dto2d, pc_im_correspondances = cam.project_on_image(X_pos = self.point_cloud_points, X_color = self.point_cloud_colors, depth_map= projection3dto2d_depth, corres=True, eps=0.12)
            
            # true_depth_map = cam.get_true_depth_map(self.point_cloud_points)
            # print('true depth map: ', true_depth_map)

            # print('true depth map max: ', true_depth_map.max())
            # print('true depth map min: ', true_depth_map.min())

            # #true depth map ok!
            # #algo esta errado cm o projection3dto2d
            # plt.figure()
            # plt.imshow(true_depth_map)
            # plt.show()

            features_interpolation = None
            pose_tot.append(pose)
            image_crop_tot.append(np.asarray(image_crop)/255.0)
            depth_tot.append(depth) #depth is the Z coordinate (1 value for each 640x480 original pixel)
            image_DINO_features_tot.append(image_DINO_features) 
            features_interpolation_tot.append(features_interpolation) #use more than 1 feature, in this case not yet
            projection3dto2d_depth_tot.append(projection3dto2d_depth)
            projection3dto2d_tot.append(projection3dto2d)
            pc_im_correspondances_tot.append(pc_im_correspondances)


            
        sample = {
            "original_image": original_image,
            'pose': pose_tot, #all TCW
            'image': image_crop_tot, #all img with 224x224
            "image_DINO_features": image_DINO_features_tot, #all (28,28,384) tuples of patch & feature
            "depth_map" : depth_tot, #each z coordinate
            'proj3dto2d_depth': projection3dto2d_depth_tot,
            "feature_interpolation": features_interpolation_tot,
            'proj3dto2d': projection3dto2d_tot,
            'correspondances' : pc_im_correspondances_tot,

        }

        return sample
