#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:01:29 2024

@author: guillaume
"""
import open3d as o3d
import numpy as np
import time

class Viewer():
    
    def __init__(self,):
        WIDTH = 1280
        HEIGHT = 720
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(width=WIDTH, height=HEIGHT)
        return


    def run (self,):
        self.viz.run()
        return
    
    def createColoredPCD(self, U, colors_U):
        pcd = o3d.geometry.PointCloud()    
        # From numpy to Open3D
        pcd.points = o3d.utility.Vector3dVector(U)
        pcd.colors = o3d.utility.Vector3dVector(colors_U)
        return pcd

    def drawCameras(self,K, imgs, Mpw, color_first=True, show_imgs=True, cam_colors=None, size=1.):
        #draw cameras
            
        line_set_list = []
        frame_list = []
        Rwc_viz_list = []
        twc_viz_list = []
        
        for c in range(len(Mpw)):
            Kinv = np.linalg.inv(K[c])
            
            wIm = imgs[c].shape[1]
            hIm = imgs[c].shape[0]
            
            #show frustum
            points_in_cam_ref = size*np.array([[0,0,0], 
                                 [0,0,1]@(Kinv.T),
                                 [wIm,0,1]@(Kinv.T), 
                                 [wIm,hIm,1]@(Kinv.T),
                                 [0,hIm,1]@(Kinv.T)])
            
            
            Rcw = Mpw[c][:3,:3]
            tcw = Mpw[c][:3,3]
            twc = -tcw@Rcw
            Rwc = Rcw.T
            
            points_in_w = (points_in_cam_ref @ Rcw) + twc
            lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]
            ]
            
            if(cam_colors!=None):
                color_cur = cam_colors[c]
            elif (c==0 and color_first == True):
                color_cur = [0, 0, 0]
            else:
                colors = [1, 0, 0]
                
            colors = [color_cur for i in range(len(lines))]
                
            line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_in_w),
            lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.viz.add_geometry(line_set)
            line_set_list.append(line_set)
            
            #show camera coordinate system
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
            frame.rotate(Rwc, np.zeros(3))
            frame.translate(twc)
            
            Rwc_viz_list.append(Rwc)
            twc_viz_list.append(twc)
    
            self.viz.add_geometry(frame)
            frame_list.append(frame)
            
            if(show_imgs == True):
                #show image
                # Define the vertices and faces for a square mesh
                #vertices = points_in_w[1:,:]
                vertices = points_in_w[[4,3,2,1],:]
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                    [2, 1, 0],
                    [3, 2, 0]
                ])
        
        
                # create the uv coordinates
                v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                                 [0, 1], [1, 0], [0, 0],
                                 [1, 0], [1, 1], [0, 1],
                                 [0, 0], [1, 0], [0, 1]])
        
                # assign the texture to the mesh
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
                mesh.textures = [o3d.geometry.Image(imgs[c].astype(np.float32))]
                mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
                mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
        
                self.viz.add_geometry(mesh)
        return line_set_list, frame_list, Rwc_viz_list, twc_viz_list

    def updateCameras(self, line_set_list, frame_list, Rwc_viz_list, twc_viz_list, K, imgs, Mpw, color_first=True, show_imgs=True, cam_colors=None):
        #draw cameras
        
        
        
        for c in range(len(line_set_list)):
            Kinv = np.linalg.inv(K[c])
            
            wIm = imgs[c].shape[1]
            hIm = imgs[c].shape[0]
            
            #show frustum
            points_in_cam_ref = [[0,0,0], 
                                 [0,0,1]@(Kinv.T),
                                 [wIm,0,1]@(Kinv.T), 
                                 [wIm,hIm,1]@(Kinv.T),
                                 [0,hIm,1]@(Kinv.T)]
            
            
            Rcw = Mpw[c][:3,:3]
            tcw = Mpw[c][:3,3]
            twc = -tcw@Rcw
            Rwc = Rcw.T
            
            points_in_w = (points_in_cam_ref @ Rcw) + twc
            
            line_set_list[c].points = o3d.utility.Vector3dVector(points_in_w)
            isUp = self.viz.update_geometry(line_set_list[c])
            assert(isUp == True)
            #line_set_list.append()
            
            #show camera coordinate system
            frame_list[c].translate(-twc_viz_list[c])
            frame_list[c].rotate(Rwc_viz_list[c].T, np.zeros(3))
            frame_list[c].rotate(Rwc, np.zeros(3))
            frame_list[c].translate(twc)
            
            Rwc_viz_list[c] = Rwc
            twc_viz_list[c] = twc
    
    
            self.viz.update_geometry(frame_list[c])
            #frame_list.append(frame)
            
            if(show_imgs == True):
                #show image
                # Define the vertices and faces for a square mesh
                #vertices = points_in_w[1:,:]
                vertices = points_in_w[[4,3,2,1],:]
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                    [2, 1, 0],
                    [3, 2, 0]
                ])
        
        
                # create the uv coordinates
                v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                                 [0, 1], [1, 0], [0, 0],
                                 [1, 0], [1, 1], [0, 1],
                                 [0, 0], [1, 0], [0, 1]])
        
                # assign the texture to the mesh
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
                mesh.textures = [o3d.geometry.Image(imgs[c].astype(np.float32))]
                mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
                mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
        
                self.viz.add_geometry(mesh)
                
        #ADD new cameras
        for c in range(len(line_set_list),len(Mpw)):
            Kinv = np.linalg.inv(K[c])
            
            wIm = imgs[c].shape[1]
            hIm = imgs[c].shape[0]
            
            #show frustum
            points_in_cam_ref = [[0,0,0], 
                                 [0,0,1]@(Kinv.T),
                                 [wIm,0,1]@(Kinv.T), 
                                 [wIm,hIm,1]@(Kinv.T),
                                 [0,hIm,1]@(Kinv.T)]
            
            
            Rcw = Mpw[c][:3,:3]
            tcw = Mpw[c][:3,3]
            twc = -tcw@Rcw
            Rwc = Rcw.T
            
            points_in_w = (points_in_cam_ref @ Rcw) + twc
            lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]
            ]
            
           
            color_cur = [0, 0, 0]
           
                
            colors = [color_cur for i in range(len(lines))]
                
            line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_in_w),
            lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.viz.add_geometry(line_set)
            line_set_list.append(line_set)
            
            #show camera coordinate system
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            frame.rotate(Rwc, np.zeros(3))
            frame.translate(twc)
    
            Rwc_viz_list.append(Rwc)
            twc_viz_list.append(twc)
    
            self.viz.add_geometry(frame)
            frame_list.append(frame)
            
            if(show_imgs == True):
                #show image
                # Define the vertices and faces for a square mesh
                #vertices = points_in_w[1:,:]
                vertices = points_in_w[[4,3,2,1],:]
                faces = np.array([
                    [0, 1, 2],
                    [0, 2, 3],
                    [2, 1, 0],
                    [3, 2, 0]
                ])
        
        
                # create the uv coordinates
                v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                                 [0, 1], [1, 0], [0, 0],
                                 [1, 0], [1, 1], [0, 1],
                                 [0, 0], [1, 0], [0, 1]])
        
                # assign the texture to the mesh
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
                mesh.textures = [o3d.geometry.Image(imgs[c].astype(np.float32))]
                mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
                mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
        
                self.viz.add_geometry(mesh)
                
        return line_set_list, frame_list, Rwc_viz_list, twc_viz_list

    def drawPointCloud(self, Uw, colors_Uw):
          #get PCL colors
         
          pcd = self.createColoredPCD(Uw,colors_Uw)
          
          #show PCL
          self.viz.add_geometry(pcd)
      
      