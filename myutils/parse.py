import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from myutils.loads import load_class

def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='Clustering script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args() #in this line args beco the file input
    assert args.file.endswith(('.yaml')), 'You need to provide a .yaml file'
    return args

def get_default_config(cfg_default):
    """Get default configuration from file"""
    paths = cfg_default.replace('[','')  #solve some erros with startwith in some compilers
    paths = cfg_default.replace('/', '.')
    config = load_class('get_cfg_defaults',
                         paths,
                         concat=False)()
    
    config.merge_from_list(['default', cfg_default])
    return config

def merge_cfg_file(config, cfg_file=None):
    """Merge config file"""
    if cfg_file is not None:
        config.merge_from_file(cfg_file)
        config.merge_from_list(['config', cfg_file])

    return config



def parse_config(cfg_default, cfg_file):
    """
    Parse model configuration for training

    Parameters
    ----------
    cfg_default : str
        Default **.py** configuration file
    cfg_file : str
        Configuration **.yaml** file to override the default parameters

    Returns
    -------
    config : CfgNode
        Parsed model configuration
    """
    config = get_default_config(cfg_default) # Loads default configuration
    config = merge_cfg_file(config, cfg_file) # Merge configuration file
    return config

def get_cfg_node(file):
    print('[retrinving file data by yaml...]')
    """
    receive a str with the name of the .yaml to train, returns the .yaml data
    needs a default_config.py

    Parameters
    ----------
    file : str
        File, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file

    Returns
    -------
    config : CfgNode
        Parsed model configuration
    """

    if file.endswith('yaml'): # If it's a .yaml configuration file
        cfg_default = 'configs/default_config'
        return parse_config(cfg_default, file) #creates based on str the path
    else: # We have a problem
        raise ValueError('You need to provide a .yaml ')


if __name__ == "__main__":

    cfg_default = 'configs/default_config'
    config = get_default_config(cfg_default)
    cfg_file = 'configs/conference_room01.yaml'
    merge_cfg_file(config, cfg_file)
    f = open(config.data.point_cloud.path, 'r')
    print(f.readline())