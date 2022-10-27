import numpy as np
import open3d as o3d
import os

FIX_COLORS = np.array([
    [0,1,0], 
    [0,0,1], 
    [1,1,0], 
    [1,0,1], 
    [1,1,0], 
    [0,1,1],
    [0.5,1,0],
    [0,0.5,1],
    [1,0,0.5],
    [1,1,0.5],
    [0.5,1,1]
    ])

#################################################################################### - saving pc
def load_point_cloud(path):
    """ load point cloud from a file.
    
        Args:
            point_cloud: o3d.geometry.PointCloud
        
        Returns:
            o3d.geometry.PointCloud
    """
    if path.endswith('.ply'):
        point_cloud_loaded = o3d.io.read_point_cloud(path)
        return point_cloud_loaded
    if path.endswith('.txt'):
        print('[reading a point cloud from a txt...]')
        point_cloud_loaded = o3d.io.read_point_cloud(path, format="xyzrgb")
        return point_cloud_loaded

def save_point_cloud(point_cloud, path):
    """ Save a point cloud to the specified path.
    
    Args:
        point_cloud: o3d.geometry.PointCloud
        path: str
        
    Returns:
        None
    """
    o3d.io.write_point_cloud(path, point_cloud)

#################################################################################### - show pc
def show_point_clouds(point_clouds):
    """ Show a list of point clouds.
    
        Args:
            point_clouds: list of o3d.geometry.PointCloud
        
        Returns:
            None
    """
    o3d.visualization.draw_geometries(point_clouds, width = 1500, height = 800)

def show_point_cloud(point_cloud,  window_name="Point Cloud vizualization", width = 1500, height = 800):
    """ Show a list of point clouds.
    
        Args:
            point_clouds: o3d.geometry.PointCloud
        
        Returns:
            None
    """

    point_clouds = [point_cloud]
    window_name = window_name
        
    o3d.visualization.draw_geometries(point_clouds, window_name, width , height)

def show_point_clouds_with_labels(point_clouds_np, labels, random_colors = False):

    print('[showing point cloud with labels...]')
    """ Show a list of point clouds with their labels.
    
        Args:
            point_clouds_np: list of numpy arrays of shape (n, d)
            labels: list of numpy arrays of shape (n,), same of points in the same cluster
        
        Returns:
            o3d.geometry.PointCloud
    """

    colors = np.random.rand(len(labels), 3) #for each pixel
    # [0.15 0.85 0.32] normalzied colors in rgb
    
    if random_colors:
        point_colors = [colors[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
    else:
        point_colors = [FIX_COLORS[labels[i]] for i in range(len(labels))] #for each point in pc (len labels)
   
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_clouds_np)
    point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
    
    show_point_cloud(point_cloud)
    return point_cloud

#################################################################################### - treat pc
def convert_xyzrgb_to_ply(path):
    print('[converting the txt file to ply...]')
    default = open(path, 'r')
    normalized_xyzrgb = open('trash/teste1.txt', 'w')
    # x, y, z, r, g, b = [float(x) for x in next(default).split()] # read first line
    array = []
    for line in default: # read rest of lines
        array.append([float(x) for x in line.split()])
    for vector in array:
        normalized_xyzrgb.write(f'{vector[0]} {vector[1]} {vector[2]} {(vector[3]/255):.3f} {(vector[4]/255):.3f} {(vector[5]/255):.3f}\n')

    

    default.close
    normalized_xyzrgb.close




if __name__ == "__main__":
    path = "data/3d/Area_1/conferenceRoom_1/conferenceRoom_1.txt"
    
    # convert_xyzrgb_to_ply(path)

    path2 = 'trash/teste1.txt'
    pcd = o3d.io.read_point_cloud(path, format="xyz")
    o3d.io.write_point_cloud("output.ply", pcd)

    pc = load_point_cloud('output.ply')


    show_point_cloud(pc)