import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import myutils.point_cloud as pc
from myutils.parse import parse_args, get_default_config, merge_cfg_file
import numpy as np

from run import run_Ayoub_adapted


if __name__ == "__main__":

    print('[main starting ...]')
    # args = parse_args()

    cfg_file = 'configs\conference_room01.yaml'
    run_Ayoub_adapted(cfg_file, number_images =10, save_pc=False, K=6)
