import numpy as np
import matplotlib.pyplot as plt
import pickle
from BA_LM_two_views_schur import BA_LM_two_views_schur
from BA_LM_schur import BA_LM_schur
from BA_LM_localization import BA_LM_localization
import utils
import time
from PIL import Image
import cv2 as cv
from os.path import join


#%%Load images, tracks, initial camera poses and 3D point cloud
with open('data_ready.pkl', 'rb') as f:
    p_list = pickle.load(f) 
    p = [np.array(pi) for pi in p_list] #kpts
   
    K = np.array(pickle.load(f)) # camera calibration matrix
   
    tracks_list = pickle.load(f) # tracks
    tracks_full = [{'p3D_keys':np.array(tracks_i['p3D_keys']),'p2D_ids':np.array(tracks_i['p2D_ids'])}  for tracks_i in tracks_list]
    
    im_names = pickle.load(f) # images names
    
    p3D_keys_to_ids = np.array(pickle.load(f))
    

#%% INITIALISATION - Start from first two images

imA_id = 0
imB_id = 1
assert(imA_id== 0 and imB_id==1)

_, idsA, idsB = np.intersect1d(tracks_full[imA_id]['p3D_keys'], tracks_full[imB_id]['p3D_keys'], assume_unique=False, return_indices=True)


I_A = np.array(Image.open(join('images',im_names[imA_id])))
I_B = np.array(Image.open(join('images',im_names[imB_id])))


kptA = cv.KeyPoint.convert(p[imA_id])
kptB = cv.KeyPoint.convert(p[imB_id])

matches = np.vstack((tracks_full[imA_id]['p2D_ids'][idsA],tracks_full[imB_id]['p2D_ids'][idsB])).astype(int)
matches_cv = [cv.DMatch(matches[0,i], matches[1,i], 0) for i in range(100)]#matches.shape[1])]


fig2, ax2 = plt.subplots(1)
fig2.suptitle('Initial pair : matches')
ax2.imshow(cv.drawMatches(I_A,kptA,I_B,kptB,matches_cv,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
plt.pause(0.01)


p_A = p[imA_id][tracks_full[imA_id]['p2D_ids'][idsA],:]
p_A_hom = np.ones((p_A.shape[0],3))
p_A_hom[:,:2] = p_A.copy()
p_B = p[imB_id][tracks_full[imB_id]['p2D_ids'][idsB],:]
p_B_hom = np.ones((p_B.shape[0],3))
p_B_hom[:,:2] = p_B.copy()

#%% INITIALISATION - Get relative pose from essential (obtained using RANSAC+five-point algorithm)

K_inv = np.linalg.inv(K)
m_B_hom = p_B_hom @ K_inv.T # passage dans le plan focal pour find essential
m_A_hom = p_A_hom @ K_inv.T
E_AB, _ = cv.findEssentialMat(m_B_hom[:,:2],m_A_hom[:,:2],method=cv.LMEDS)
                   
nInliersChirality, R_BA, t_BA, maskChirality_cv = cv.recoverPose(E_AB.T, p_A_hom[:,:2], p_B_hom[:,:2], K)
maskChirality = (maskChirality_cv!=0).flatten() # maskChirality : info sur points devant caméra
t_BA = t_BA.flatten()

M_BA = np.eye(4) # rassemble t_BA et R_BA dans une meme matrice de Projection qui va du repere B a A
M_BA[:3,:3] = R_BA
M_BA[:3,3] = t_BA

p_A_filter = p_A_hom[maskChirality,:]
p_B_filter = p_B_hom[maskChirality,:]

# comme repere A = repere monde (W) alors M_BA est matrice qui passe de B a repère Monde aussi 
P_AW = K @ np.hstack((np.eye(3), np.zeros((3,1)))) # matrice extrinseque alignee avec repere mondiale
P_BW = K @ M_BA[:3,:] 
U_A_hom = cv.triangulatePoints(P_AW, P_BW, p_A_filter[:,:2].T, p_B_filter[:,:2].T)
U_A = (U_A_hom[:3,:]/U_A_hom[3,:]).T
assert (U_A[:,2] > 0).all() #all 3D points should have a positive depth in A

# tentative de reprojection basee sur les points 3D obtenus avec la matrice essentielle & triangulation
p_A_pred = (U_A/U_A[:,2:3]) @ K.T
p_B_pred = ((U_B := (U_A @ R_BA.T) + t_BA)/U_B[:,2:3]) @ K.T

reproj_error_A = (1/p_A_pred.shape[0])*np.sqrt(((p_A_pred - p_A_filter)**2).sum(axis=1)).sum()
reproj_error_B = (1/p_B_pred.shape[0])*np.sqrt(((p_B_pred - p_B_filter)**2).sum(axis=1)).sum()

fig3, axs3 = plt.subplots(ncols=2)
fig3.suptitle('Initial pair : reproj error before BA')
axs3[0].imshow(I_A)
axs3[0].scatter(p_A_filter[:,0], p_A_filter[:,1], marker ='o', facecolors='none', edgecolors='r')
axs3[0].scatter(p_A_pred[:,0], p_A_pred[:,1], marker ='x', color='b')
axs3[0].plot(np.vstack((p_A_filter[:,0],p_A_pred[:,0])), np.vstack((p_A_filter[:,1],p_A_pred[:,1])))
axs3[0].set_title('Reproj. err. {0:0.2f} pix'.format(reproj_error_A))
axs3[1].imshow(I_B)
axs3[1].scatter(p_B_filter[:,0], p_B_filter[:,1], marker ='o', facecolors='none', edgecolors='r')
axs3[1].scatter(p_B_pred[:,0], p_B_pred[:,1], marker ='x', color='b')
axs3[1].plot(np.vstack((p_B_filter[:,0],p_B_pred[:,0])), np.vstack((p_B_filter[:,1],p_B_pred[:,1])))
axs3[1].set_title('Reproj. err. {0:0.2f} pix'.format(reproj_error_B))

#%% INITIALISATION - Create tracks of 3D points actually reconstructed
n_Uw = 0
tracks =[]
tracks.append({'p3D_keys':tracks_full[imA_id]['p3D_keys'][idsA[maskChirality]],'p2D_ids':tracks_full[imA_id]['p2D_ids'][idsA[maskChirality]]})
tracks.append({'p3D_keys':tracks_full[imB_id]['p3D_keys'][idsB[maskChirality]],'p2D_ids':tracks_full[imB_id]['p2D_ids'][idsB[maskChirality]]})
n_Uw = len(tracks[imA_id]['p3D_keys'])
p3D_keys_to_ids[tracks[imA_id]['p3D_keys']] = np.arange(n_Uw,dtype=np.int64) #vector that maps keys to ids in the list of 3D points that are reconstructed
p3D_keys_reconstructed = tracks[imA_id]['p3D_keys']


#%% INITIALISATION - Bundle Adjustment

# calcul erreure de reprojection après avoir estimé les Rotations et translations puis triangulé
# Pour calculer optimiser les paramètres (U, t, R) de cette erreure de reprojection on utilise 
# moindre carrés linéaires. Pour résoudre cela on a besoin d'une bonne initialisation qui se fait 
# d'abord avec l'estimation de la matrice Essentielle en geom epipolaire comme fait au dessus.
# Puis on linéarise la le pb d'optimisation, puis on résoud avec Levenberg Marquardt


I = [I_A, I_B]
Mwc = [np.eye(4), np.linalg.inv(M_BA)] # matrices de passage caméra vers word (eye(4) est l'origine (caméra A))
Uw = U_A.copy() # replacer les premiers points 3D issus 

Mwc, Uw = utils.normalizeReconstructionScale(Mwc,Uw)

nCam = len(Mwc)

# calcul des erreures de reprojection avant de commencer le Bundle Adjustment 
fig4, axs4 = plt.subplots(ncols=nCam)
fig4.suptitle('Initial pair : reproj error before BA')
for cam in range(nCam):
    
    assert((tracks[cam]['p3D_keys']>0).all())
    Mwc_cur = Mwc[cam]
    U_c = (Uw[p3D_keys_to_ids[tracks[cam]['p3D_keys']],:] - Mwc_cur[:3,3])@Mwc_cur[:3,:3]
    p_c_pred = (U_c/U_c[:,2:3]) @ K.T
    
    p_c = p[cam][tracks[cam]['p2D_ids'],:]

    assert(np.all(U_c[:,2] > 0.)) #all 3D points observed in an image should have positive depth
    if(cam == 0):
        assert(np.allclose(p_c_pred, p_A_pred))
        assert(np.allclose(p_c, p_A_filter[:,:2]))
    elif  (cam==1):
        assert(np.allclose(p_c_pred, p_B_pred))
        assert(np.allclose(p_c, p_B_filter[:,:2]))
        
     
    reproj_error = (1/p_c_pred.shape[0])*np.sqrt(((p_c_pred[:,:2] - p_c)**2).sum(axis=1)).sum()
    
    axs4[cam].imshow(I[cam])
    axs4[cam].scatter(p_c[:,0], p_c[:,1], marker ='o', facecolors='none', edgecolors='r')
    axs4[cam].scatter(p_c_pred[:,0], p_c_pred[:,1], marker ='x', color='b')
    axs4[cam].plot(np.vstack((p_c[:,0],p_c_pred[:,0])), np.vstack((p_c[:,1],p_c_pred[:,1])))
    axs4[cam].set_title('Reproj. err. {0:0.2f} pix'.format(reproj_error))

plt.pause(0.1)


BA = BA_LM_two_views_schur(Mwc, Uw, p3D_keys_to_ids, tracks, K, p)
#BA = BA_LM_two_views(Mwc, Uw, p3D_keys_to_ids, tracks, K, p)
begin = time.time()
BA.optimize()
print('{} sec'.format(time.time()-begin))
Mwc = BA.getPoses()
Uw = BA.getPointCloud()

fig5, axs5 = plt.subplots(ncols=nCam)
fig5.suptitle('Initial pair : reproj error before BA')
for cam in range(nCam):
    Mwc_cur = Mwc[cam]
    U_c = (Uw[p3D_keys_to_ids[tracks[cam]['p3D_keys']],:] - Mwc_cur[:3,3])@Mwc_cur[:3,:3]
    p_c_pred = (U_c/U_c[:,2:3]) @ K.T
    p_c = p[cam][tracks[cam]['p2D_ids'],:]
     
    assert(np.all(U_c[:,2] > 0.)) #all 3D points observed in an image should have positive depth
    
    reproj_error = (1/p_c_pred.shape[0])*np.sqrt(((p_c_pred[:,:2] - p_c)**2).sum(axis=1)).sum()
    
    
    axs5[cam].imshow(I[cam])
    axs5[cam].scatter(p_c[:,0], p_c[:,1], marker ='o', facecolors='none', edgecolors='r')
    axs5[cam].scatter(p_c_pred[:,0], p_c_pred[:,1], marker ='x', color='b')
    axs5[cam].plot(np.vstack((p_c[:,0],p_c_pred[:,0])), np.vstack((p_c[:,1],p_c_pred[:,1])))
    axs5[cam].set_title('Reproj. err. {0:0.2f} pix'.format(reproj_error))

plt.pause(0.1)


fig6, axs6 = plt.subplots(ncols=2)
fig6.suptitle('Current image : reproj error localization')

#%%

for new_im_id in range(2, len(im_names)):
    #%% Add image (initialize for a new image)
    
    pass 
    # search keys of 3D points already reconstructed and seen in the new image -> np.intersect1d
    _, idsReconstruct, idsNew = np.intersect1d(p3D_keys_reconstructed, tracks_full[new_im_id]['p3D_keys'], assume_unique=False, return_indices=True)
    
    # remove 3D points seen in the new image that have a negative depth before localisation
    # cette étape peut être ignorée pour l'instant et reconstruire les points 3D même si ils se trouvent être à l'arriere de la caméra
    
    # all 3D points seen in the new image should have positive depth ->  assert(np.all(U_c[:,2] > 0.)) 
    
    
    I_c= np.array(Image.open(join('images',im_names[new_im_id]))) #load new image for visualization and debug
    I.append(I_c) # store new image for visualization and debug
    
    #Visualize reprojection error before localization
    
    # initialiser les paramètres pour la nouvelle image et en déduire une première reprojection 
    # comme cela avait été fait dans la première partie pour la première caméra.
    
    
    # axs6[0].clear()
    # plt.pause(0.1)
    # axs6[0].imshow(I[new_im_id])
    # axs6[0].scatter(p_c_loc[:,0], p_c_loc[:,1], marker ='o', facecolors='none', edgecolors='r')
    # axs6[0].scatter(p_c_loc_pred[:,0], p_c_loc_pred[:,1], marker ='x', color='b')
    # axs6[0].plot(np.vstack((p_c_loc[:,0],p_c_loc_pred[:,0])), np.vstack((p_c_loc[:,1],p_c_loc_pred[:,1])))
    # axs6[0].set_title('Loc. Init. Reproj. err. {0:0.2f} pix'.format(reproj_error))
    # plt.pause(0.1)
    
    #%% LOCALISATION
    
    
    
    pass 
    #Visualize reprojection error after localization
    # axs6[1].clear()
    # plt.pause(0.1)
    # axs6[1].imshow(I[new_im_id])
    # axs6[1].scatter(p_c_loc[:,0], p_c_loc[:,1], marker ='o', facecolors='none', edgecolors='r')
    # axs6[1].scatter(p_c_loc_pred[:,0], p_c_loc_pred[:,1], marker ='x', color='b')
    # axs6[1].plot(np.vstack((p_c_loc[:,0],p_c_loc_pred[:,0])), np.vstack((p_c_loc[:,1],p_c_loc_pred[:,1])))
    # axs6[1].set_title('Loc. Refined. Reproj. err. {0:0.2f} pix'.format(reproj_error))
    # plt.pause(0.1)
    

    #%% TRIANGULATION of new 3D points
    
    pass
    #find 3D keys points seen in the image new image that are not already reconstructed
    
    print('Number of 3D points before adding new: {}'.format(len(Uw)))
        
    #tracks.append(tracks_new)
    
    #%% Bundle adjustment
    
    pass
    Mwc, Uw = utils.normalizeReconstructionScale(Mwc,Uw)
    
#%% Save reconstruction
with open('final_reconstruction.pkl', 'wb') as f:
    p_list = [pi.tolist() for pi in p] 
    pickle.dump(p_list, f)

    pickle.dump(K.tolist(), f)

    Mwc_list = [Mwi.tolist() for Mwi in Mwc] 
    pickle.dump(Mwc_list, f)
    
    pickle.dump(Uw, f)
    
    tracks_list = [{'p3D_keys':tracks_i['p3D_keys'].tolist(),'p2D_ids':tracks_i['p2D_ids'].tolist()}  for tracks_i in tracks]
    pickle.dump(tracks_list, f)
    
    pickle.dump(im_names, f)
    
    pickle.dump(p3D_keys_to_ids.tolist(), f)
print('3D model saved')
