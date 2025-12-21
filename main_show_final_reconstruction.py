#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:36:57 2024

@author: guillaume
"""

import numpy as np
import pickle
import viewer

from PIL import Image
from os.path import join

track_length_min = 2 #set to 4 to remove bad 3D points

#%% Load images, tracks, initial camera poses and 3D point cloud
with open('final_reconstruction.pkl', 'rb') as f:
    p_list = pickle.load(f) 
    p = [np.array(pi) for pi in p_list] #kpts
   
    K = np.array(pickle.load(f)) # camera calibration matrix
   
    Mwc_list = pickle.load(f)#camera poses
    Mwc = [np.array(Mwc_cur) for Mwc_cur in Mwc_list]
    
    Uw = pickle.load(f) #3D point cloud
    
    tracks_list = pickle.load(f) # tracks
    tracks = [{'p3D_keys':np.array(tracks_i['p3D_keys']),'p2D_ids':np.array(tracks_i['p2D_ids'])}  for tracks_i in tracks_list]
    
    im_names = pickle.load(f) # images names
    
    p3D_keys_to_ids = np.array(pickle.load(f))
    
    
I = [np.array(Image.open(join('images',im_name))).astype(float)/255 for im_name in im_names]

#%% Find color of 3D points
colors_U = np.zeros_like(Uw)
track_length = np.zeros((Uw.shape[0]))

for cam in range(len(Mwc)):
    Mwc_cur = Mwc[cam]
    U_c = (Uw[p3D_keys_to_ids[tracks[cam]['p3D_keys']],:] - Mwc_cur[:3,3])@Mwc_cur[:3,:3]
    p_c_pred = ((U_c/U_c[:,2:3]) @ K.T).astype(int)    
    
    colors_U[p3D_keys_to_ids[tracks[cam]['p3D_keys']],:] = (I[cam][p_c_pred[:,1],p_c_pred[:,0],:])
    track_length[p3D_keys_to_ids[tracks[cam]['p3D_keys']]] += 1

#%% Remove 3D points that are too far from the origin and have a small track length
mask_Uw = (np.sqrt((Uw**2).sum(axis=1))<30) * (track_length >= track_length_min)
Uw_masked = Uw[mask_Uw,:]
colors_U_masked = colors_U[mask_Uw,:]

#%% Visualize
Mcw = [np.linalg.inv(M) for M in Mwc]
K_list = [K]*len(Mcw)
viewer = viewer.Viewer()
line_set_list, frame_list, Rwc_viz_list, twc_viz_list = viewer.drawCameras(K_list, I, Mcw, color_first=True, show_imgs=True, cam_colors=None, size=0.2)
viewer.drawPointCloud(Uw_masked, colors_U_masked)

viewer.run()
    
