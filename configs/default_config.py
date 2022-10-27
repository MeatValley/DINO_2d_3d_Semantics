from yacs.config import CfgNode as CN
from datetime import datetime

########################################################################################################################
cfg = CN()
cfg.proj_path = ''
########################################################################################################################
### EXPERIMENT
########################################################################################################################
cfg.experiment = CN()
cfg.experiment.name = "default_name"
cfg.experiment.area = 'ab'
cfg.experiment.time = datetime.now().strftime(f'%Y-%m-%d-%H-%M')
########################################################################################################################
### PIPELINE
########################################################################################################################
cfg.pipeline = CN()
cfg.pipeline.clustering = CN()
cfg.pipeline.clustering.algo = 'kmeans'
cfg.pipeline.clustering.k = 2
cfg.pipeline.clustering.init_centroids = '++'
cfg.pipeline.feature_extractor = CN()
cfg.pipeline.feature_extractor.network = 'DINO'
cfg.pipeline.feature_extractor.model = "dino_deitsmall8_pretrain.pth"
########################################################################################################################
### DATA
########################################################################################################################
cfg.data = CN()
cfg.data.name = ["INDOOR"]
cfg.data.path = "data/scenes/scene0000_00/"
cfg.data.split = None
cfg.data.point_cloud_name = "scene0000_00_vh_clean_2.ply"
cfg.data.transform = None
cfg.data.point_cloud = CN()
cfg.data.point_cloud.path = 'data/3d/Area_1/conferenceRoom_1/conferenceRoom_1.txt'
cfg.data.images = CN()
cfg.data.images.path = 'data/images'
########################################################################################################################
### SAVE
########################################################################################################################
cfg.save = CN()
cfg.save.folder = "configs/logs/"
cfg.save.point_cloud = True
cfg.save.images = True
########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.config = ''                 # Run configuration file
cfg.default = ''                # Run default configuration file
cfg.save.pretrained = ''        # Pretrained checkpoint
cfg.prepared = False   


def get_cfg_defaults():
    return cfg.clone()
