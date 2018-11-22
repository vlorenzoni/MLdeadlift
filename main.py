import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.motion_data as mmd

import utils.valerio_model as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable


motion_loading=mmd.motion_load_IPEM
motion_names=mmd.bike_names[:3]
motion_dict=dict([(motion_name,motion_loading(motion_name,skip_rows=10,skip_columns=2)) for motion_name in motion_names])
motions_data =[mmd.center_norm_data(md[0]) for md in motion_dict.values()]
motions_data=[md.reshape(md.shape[0],-1) for md in motions_data]
