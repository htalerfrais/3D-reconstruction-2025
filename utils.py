#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:51:39 2024

@author: guillaume
"""
import numpy as np

def normalizeReconstructionScale(Mwc,Uw):
    
    nCam = len(Mwc)
    
    twc = np.array([M[:3,3] for M in Mwc])
    
    #centering around first camera
    twc_centered = twc - twc[0,:]
    Uw_centered = Uw - twc[0,:]
    
    #second camera sould be at a distance of 1
    norm_tw1 = np.sqrt(twc_centered[1,:].dot(twc_centered[1,:]))
    twc_norm = twc_centered/norm_tw1
    Uw_norm = Uw_centered/norm_tw1
    
    for i in range(nCam):
        Mwc[i][:3,3] = twc_norm[i,:]
        
    return Mwc, Uw_norm
        
        
def checkNegativeDepth(Mwc, Uw, p3D_keys_to_ids, tracks, K, p):
    
    nCam = len(Mwc)
    
    for cam in range(nCam):
        Mwc_cur = Mwc[cam]
        U_c = (Uw[p3D_keys_to_ids[tracks[cam]['p3D_keys']],:] - Mwc_cur[:3,3])@Mwc_cur[:3,:3]
        
        mask_pos_depth = U_c[:,2] > 0.
        
        if(~mask_pos_depth.all()):
            print('WARNING: {} PTS WITH NEGATIVE DEPTH'.format(sum(~mask_pos_depth)))
 
        return