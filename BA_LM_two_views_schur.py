import sys
import numpy as np
import scipy as sp
import math
from scipy.spatial.transform import Rotation as R
import copy

class BA_LM_two_views_schur():
    
    def __init__(self,  Mwc_init, Uw_init, p3D_keys_to_ids, tracks, K, p, maxIt=1000, lambdaMin=1e-5, lambdaMax=1e6, lambdaInit = 1e-3):
        self.lambdaInit = lambdaInit
        self.lambdaMin = lambdaMin
        self.lambdaMax = lambdaMax
        self.maxIt = maxIt
        
        self.K = K.copy()
        self.p = copy.deepcopy(p)
        self.tracks = copy.deepcopy(tracks)
        self.p3D_keys_to_ids = p3D_keys_to_ids.copy()
        
        if(len(Mwc_init)!=2):
            print("Only two view BA is implemented")
            sys.exit()
            
        self.nCam = len(Mwc_init)
        
        self.Rwc = [Mwc_init[i][:3,:3].copy() for i in range(self.nCam)]
        self.twc = [Mwc_init[i][:3,3].copy() for i in range(self.nCam)] 
        self.Uw = Uw_init.copy()
        self.nP3D = self.Uw.shape[0]
        

        self.lambda_cur = lambdaInit
        

        self.H_UU = np.zeros((self.nP3D,3,3)) #store each bloc of H_UU since H_UU is bloc-diagonal
        self.H_UU_plus_lambda = np.zeros_like(self.H_UU)
        self.H_UU_plus_lambda_inv = np.zeros_like(self.H_UU)
        
        self.H_PP = np.zeros((6*self.nCam,6*self.nCam))
        
        self.H_UP = np.zeros((3*self.nP3D,6*self.nCam)) #still memory inefficient, should not store this matrix when nP3D or nCam are not small
        self.H_PU_H_UU_plus_lambda_inv = np.zeros((6*self.nCam,3*self.nP3D))
        
        self.b_U = np.zeros((3*self.nP3D))
        self.b_P = np.zeros((6*self.nCam))

    def getPoses(self):
        Mwc = []
        for i in range(self.nCam):
            Mwc_temp = np.eye(4)
            Mwc_temp[:3,:3] = self.Rwc[i]
            Mwc_temp[:3,3] = self.twc[i]
            Mwc.append(Mwc_temp)
        return Mwc
    
    def getPointCloud(self):
        return self.Uw
    
    def compute_residuals(self):
                    
        r_list = []
        for i in range(self.nCam):
            p_vis = self.p[i][self.tracks[i]['p2D_ids'],:]

            Ui = (self.Uw[self.p3D_keys_to_ids[self.tracks[i]['p3D_keys']],:] - self.twc[i])@(self.Rwc[i])
            p_vis_pred_hom = (Ui/Ui[:,2:3]) @ self.K.T
            p_vis_pred = p_vis_pred_hom[:,:2]
            
            r_list.append(p_vis[:,:2].flatten() - p_vis_pred.flatten())

        r = np.hstack(r_list)
        return r

    def compute_cost(self):
        
        r = self.compute_residuals()
        c = (r**2).sum()

        n_res= sum(self.tracks[i]['p2D_ids'].shape[0] for i in range(self.nCam))
        c_in_pix = math.sqrt(c/n_res); #RMSE in pixels
        return c_in_pix
    
    def generators_SO3(self):
        Gx = np.array([[0.,  0., 0.],[0., 0., -1.],[ 0., 1., 0.]])
        Gy = np.array([[0.,  0., 1.],[0., 0.,  0.],[-1., 0., 0.]])
        Gz = np.array([[0., -1., 0.],[1., 0.,  0.],[ 0., 0., 0.]])
        return Gx, Gy, Gz
    
    
    def expSO3(self, w): #matrix exponential for 3D roations
        return R.from_rotvec(w).as_matrix()
    
    def compute_decomposed_linear_system(self, r):
                   
       Gx, Gy, Gz = self.generators_SO3()
       
       self.H_UU.fill(0.)
       self.H_PP.fill(0.)
       self.H_UP.fill(0.)
       self.b_U.fill(0.)
       self.b_P.fill(0.)

       l=0
       for i in range(self.nCam): #for each camera
         
           Ui = (self.Uw[self.p3D_keys_to_ids[self.tracks[i]['p3D_keys']],:] - self.twc[i])@(self.Rwc[i])
           
           nb_Pt_in_current_cam = len(self.tracks[i]['p3D_keys'])

           for t in range(nb_Pt_in_current_cam): #for each 2D point
               
               A = np.array([[1/Ui[t,2,], 0, -Ui[t,0]/(Ui[t,2]**2)], [0, 1/Ui[t,2], -Ui[t,1]/(Ui[t,2]**2)], [0, 0, 0]])
               
               #3D point derivative
               J_U = self.K[:2,:]@A@self.Rwc[i].T
               self.H_UU[self.p3D_keys_to_ids[self.tracks[i]['p3D_keys']][t],:,:] += J_U.T @ J_U
               
               #rotation derivative
               J_R = np.vstack((self.K[:2,:]@A@Gx.T@(Ui[t,:].T), self.K[:2,:]@A@Gy.T@Ui[t,:].T, self.K[:2,:]@A@Gz.T@Ui[t,:])).T
               #translation derivative
               J_T = -self.K[:2,:]@A@self.Rwc[i].T
               J_P = np.zeros((2,6))
               J_P[:,:3] = J_R
               J_P[:,3:] = J_T
               self.H_PP[i*6:(i+1)*6,i*6:(i+1)*6] += J_P.T @ J_P
               
               self.H_UP[3*self.p3D_keys_to_ids[self.tracks[i]['p3D_keys']][t]:3*(self.p3D_keys_to_ids[self.tracks[i]['p3D_keys']][t]+1),i*6:(i+1)*6] += J_U.T @ J_P
               
               self.b_U[3*self.p3D_keys_to_ids[self.tracks[i]['p3D_keys']][t]:3*(self.p3D_keys_to_ids[self.tracks[i]['p3D_keys']][t]+1)] += J_U.T @ r[l:l+2]
               self.b_P[i*6:(i+1)*6] += J_P.T @ r[l:l+2]
               
               l = l+2

       return

    
    def compute_H_UU_plus_lambda_inv(self):
        #compute inverse of each bloc of H_UU + lambda
        for i in range(self.nP3D):
            self.H_UU_plus_lambda[i,:,:] = self.H_UU[i,:,:] + self.lambda_cur*np.eye(3)
        
        self.H_UU_plus_lambda_inv = np.linalg.inv(self.H_UU_plus_lambda)
        return 

    def compute_H_PU_H_UU_plus_lambda_inv(self):
        
        for i in range(self.nP3D):
            self.H_PU_H_UU_plus_lambda_inv[:,3*i:3*(i+1)] = self.H_UP[3*i:3*(i+1),:].T @ self.H_UU_plus_lambda_inv[i,:,:]        
        return
    
    
    
    def updateParameters(self, r, c_prev, iteration):
        
        success = False
        
        while(self.lambda_cur<self.lambdaMax):
            
            
            self.compute_H_UU_plus_lambda_inv()
            self.compute_H_PU_H_UU_plus_lambda_inv()
            
            A = self.H_PP + self.lambda_cur*np.eye(6*self.nCam) - self.H_PU_H_UU_plus_lambda_inv @ self.H_UP
            d = self.b_P - self.H_PU_H_UU_plus_lambda_inv @ self.b_U
            delta_P = sp.linalg.solve(A, d,assume_a='sym')
            
            delta_U = np.zeros((3*self.nP3D))
            for i in range(self.nP3D):
                delta_U[3*i:3*(i+1)] = self.H_UU_plus_lambda_inv[i,:,:]@(self.b_U[3*i:3*(i+1)] - self.H_UP[3*i:3*(i+1),:]@delta_P) 
            
            #update variables
            Uw_prev = self.Uw.copy()
            self.Uw  += np.reshape(delta_U,(self.nP3D,3))
            
            Rwc_prev = copy.deepcopy(self.Rwc)
            twc_prev = copy.deepcopy(self.twc)
            for cam in range(self.nCam):                
                self.Rwc[cam] = self.Rwc[cam]@self.expSO3(delta_P[6*cam:3+6*cam])
                self.twc[cam] += delta_P[3+6*cam:6+6*cam]
      
            #compute cost
            c_new = self.compute_cost()
            print('Iter:{} Error:{} pix Lambda:{}'.format(iteration,c_new,self.lambda_cur))

            if(c_new+1e-5) < c_prev:
                success = True
                #print('c new {}, c prev {} ,success {}'.format(c_new, c_prev, success))
                if self.lambda_cur>self.lambdaMin :
                    self.lambda_cur /= 2
                break
            else:
                self.lambda_cur *= 2
                self.Rwc = Rwc_prev
                self.twc = twc_prev
                self.Uw = Uw_prev

        if not success:
            c_new = c_prev
        
        return c_new, success

    def optimize(self):
        
        
        #compute inital cost
        c = self.compute_cost()
    
        print('Iter:{} Error:{} pix'.format(0,c))
                
        for i in range(self.maxIt):
            
            #compute Jacobian and residuals
            r = self.compute_residuals()
            self.compute_decomposed_linear_system(r)
            
            #update parameters
            c, success = self.updateParameters(r, c, i)
            
            if not success:
                break  

        return
