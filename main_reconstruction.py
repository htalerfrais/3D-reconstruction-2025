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
    
t0 = time.time()

#%% INITIALISATION - Start from first two images

reproj_err_list = []
reproj_err_global_ba_list = []

imA_id = 0
imB_id = 1
assert(imA_id==0 and imB_id==1)

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

#%% INITIALISATION - Get relative pose from essential (obtained using RANSAC + five-point algorithm)

K_inv = np.linalg.inv(K)
m_B_hom = p_B_hom @ K_inv.T
m_A_hom = p_A_hom @ K_inv.T
E_AB, _ = cv.findEssentialMat(m_B_hom[:,:2],m_A_hom[:,:2],method=cv.LMEDS)
                   
nInliersChirality, R_BA, t_BA, maskChirality_cv = cv.recoverPose(E_AB.T, p_A_hom[:,:2], p_B_hom[:,:2], K)
maskChirality = (maskChirality_cv!=0).flatten()
t_BA = t_BA.flatten()

M_BA = np.eye(4)
M_BA[:3,:3] = R_BA
M_BA[:3,3] = t_BA

p_A_filter = p_A_hom[maskChirality,:]
p_B_filter = p_B_hom[maskChirality,:]

P_AW = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P_BW = K @ M_BA[:3,:]
U_A_hom = cv.triangulatePoints(P_AW, P_BW, p_A_filter[:,:2].T, p_B_filter[:,:2].T)
U_A = (U_A_hom[:3,:]/U_A_hom[3,:]).T
assert (U_A[:,2] > 0).all() #all 3D points should have a positive depth in A

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

I = [I_A, I_B]
Mwc = [np.eye(4), np.linalg.inv(M_BA)]
Uw = U_A.copy()

Mwc, Uw = utils.normalizeReconstructionScale(Mwc,Uw)

nCam = len(Mwc)

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
fig5.suptitle('Initial pair : reproj error after BA')
for cam in range(nCam):
    Mwc_cur = Mwc[cam]
    U_c = (Uw[p3D_keys_to_ids[tracks[cam]['p3D_keys']],:] - Mwc_cur[:3,3])@Mwc_cur[:3,:3]
    p_c_pred = (U_c/U_c[:,2:3]) @ K.T
    p_c = p[cam][tracks[cam]['p2D_ids'],:]
     
    assert(np.all(U_c[:,2] > 0.)) #all 3D points observed in an image should have positive depth
    
    reproj_error = (1/p_c_pred.shape[0])*np.sqrt(((p_c_pred[:,:2] - p_c)**2).sum(axis=1)).sum()
    reproj_err_list.append(reproj_error)
    
    axs5[cam].imshow(I[cam])
    axs5[cam].scatter(p_c[:,0], p_c[:,1], marker ='o', facecolors='none', edgecolors='r')
    axs5[cam].scatter(p_c_pred[:,0], p_c_pred[:,1], marker ='x', color='b')
    axs5[cam].plot(np.vstack((p_c[:,0],p_c_pred[:,0])), np.vstack((p_c[:,1],p_c_pred[:,1])))
    axs5[cam].set_title('Reproj. err. {0:0.2f} pix'.format(reproj_error))

plt.pause(0.1)


fig6, axs6 = plt.subplots(ncols=2)
fig6.suptitle('Current image : reproj error localization')


for new_im_id in range(2, len(im_names)):        #len(im_names)
    print('Image Index : ', new_im_id)
    #% Add image
     
    #search keys of 3D points already reconstructed and seen in the new image -> np.intersect1d
    common_keys, idsReconstruct, idsNew = np.intersect1d(p3D_keys_reconstructed, tracks_full[new_im_id]['p3D_keys'], assume_unique=False, return_indices=True)

    #remove 3D points seen in the new image that have a negative depth before localisation
    Mwc_guess = Mwc[-1]         #approx new camera position by previous one
    Rwc_guess = Mwc_guess[:3, :3]
    twc_guess = Mwc_guess[:3, 3]
    
    Xw = Uw[p3D_keys_to_ids[common_keys]]  #3D points in world ref common between new image and reconstructed 3D points
    x2d = p[new_im_id][tracks_full[new_im_id]['p2D_ids'][idsNew], :]        #corresponding 2D points
    U_c = (Xw - twc_guess) @ Rwc_guess          # project 3D points to new camera referential
    
    mask = U_c[:, 2] > 1e-6       #mask to keep only points minimal distance front camera
    U_c = U_c[mask]             #apply mask
    Xw = Xw[mask]
    p_c_loc = x2d[mask]
    
    p_c_loc_pred = (U_c / U_c[:, 2:3]) @ K.T    #get the projected 2D points of the 3D points into Mwc_guess
    p_c_loc_pred = p_c_loc_pred[:, :2]
    
    #all 3D points seen in the new image should have positive depth ->  assert(np.all(U_c[:,2] > 0.)) 
    assert(np.all(U_c[:,2] > 0.))
    
    I_c= np.array(Image.open(join('images',im_names[new_im_id]))) #load new image for visualization and debug
    I.append(I_c) # store new image for visualization and debug
    
    reproj_error = (1/p_c_loc_pred.shape[0])*np.sqrt(((p_c_loc_pred[:,:2] - p_c_loc)**2).sum(axis=1)).sum()

    #Visualize reprojection error before localization
    axs6[0].clear()
    plt.pause(0.1)
    axs6[0].imshow(I[new_im_id])
    axs6[0].scatter(p_c_loc[:,0], p_c_loc[:,1], marker ='o', facecolors='none', edgecolors='r')
    axs6[0].scatter(p_c_loc_pred[:,0], p_c_loc_pred[:,1], marker ='x', color='b')
    axs6[0].plot(np.vstack((p_c_loc[:,0],p_c_loc_pred[:,0])), np.vstack((p_c_loc[:,1],p_c_loc_pred[:,1])))
    axs6[0].set_title('Loc. Init. Reproj. err. {0:0.2f} pix'.format(reproj_error))
    plt.pause(0.25)
    
    
    #%% LOCALISATION
    
    idsNew = idsNew[mask]   # mask contains 3D points in new cam referential that are in front of cam
    # idsNew contains the indices of the new image 3D points that are in common with reconstructed
    p_loc = [p[new_im_id]]  # contains the coordinates of all 2D points of new image 
    
    local_p3D_keys_to_ids = np.arange(len(Xw))
    
    tracks_loc = [{
        'p3D_keys': local_p3D_keys_to_ids,
        'p2D_ids': tracks_full[new_im_id]['p2D_ids'][idsNew]
    }]
    
    assert len(Xw) == len(tracks_loc[0]['p2D_ids'])
    
    BA_loc = BA_LM_localization(
        [Mwc_guess],
        Xw,
        local_p3D_keys_to_ids,
        tracks_loc,
        K,
        p_loc
    )

    BA_loc.optimize()
    Uw_new = BA_loc.getPointCloud()
    assert(np.allclose(Xw, Uw_new))
    
    Mwc_new = BA_loc.getPoses()[0]
    
    Mwc.append(Mwc_new)
    
    Rwc = Mwc_new[:3, :3]
    twc = Mwc_new[:3, 3]
    
    U_c = (Xw - twc) @ Rwc
    p_c_loc_pred = (U_c / U_c[:, 2:3]) @ K.T
    p_c_loc_pred = p_c_loc_pred[:, :2]
    
    assert(np.all(U_c[:,2] > 0.))
    
    reproj_error = np.mean(
        np.linalg.norm(p_c_loc_pred - p_c_loc, axis=1)
    )
    reproj_err_list.append(reproj_error)
    
    print(f"Localization reprojection error: {reproj_error:.2f} px")
         
    #Visualize reprojection error after localization
    axs6[1].clear()
    plt.pause(0.1)
    axs6[1].imshow(I[new_im_id])
    axs6[1].scatter(p_c_loc[:,0], p_c_loc[:,1], marker ='o', facecolors='none', edgecolors='r')
    axs6[1].scatter(p_c_loc_pred[:,0], p_c_loc_pred[:,1], marker ='x', color='b')
    axs6[1].plot(np.vstack((p_c_loc[:,0],p_c_loc_pred[:,0])), np.vstack((p_c_loc[:,1],p_c_loc_pred[:,1])))
    axs6[1].set_title('Loc. Refined. Reproj. err. {0:0.2f} pix'.format(reproj_error))
    plt.pause(0.25)
    

    #%% TRIANGULATION of new 3D points
    
    
    # find 3D keys points seen in the image new image that are not already reconstructed
    # --- CONFIGURATION ---
    min_parallax_angle = 5.0  # Seuil minimal en degrés pour accepter un point
    new_points_3D = []
    keys_actually_added = []
    
    # Identifier les points orphelins (clés qui ne sont pas encore reconstruites)
    new_keys = np.setdiff1d(tracks_full[new_im_id]['p3D_keys'], p3D_keys_reconstructed)
    
    # Pré-calculer les centres des caméras et les matrices de projection
    # Mwc[cam][:3, 3] centre de la caméra dans le monde (C)
    # Mwc[cam][:3, :3] est la rotation Monde vers Caméra (R)
    cams_centers = [m[:3, 3] for m in Mwc]
    cams_P = [K @ np.hstack((m[:3, :3].T, (-m[:3, :3].T @ m[:3, 3]).reshape(3,1))) for m in Mwc]
    
    # Créer une table d'inversion pour savoir quelle caméra a vu quelle clé (orpheline)
    # Cela évite de boucler inutilement sur toutes les caméras pour chaque point
    key_to_prev_cams = {key: [] for key in new_keys}
    for cam_idx in range(new_im_id):
        common = np.intersect1d(tracks_full[cam_idx]['p3D_keys'], new_keys)
        for k in common:
            key_to_prev_cams[k].append(cam_idx)
    
    # Traitement point par point
    C_new = cams_centers[new_im_id]
    P_new = cams_P[new_im_id]
    p_new_all = p[new_im_id] # Tous les points 2D de la nouvelle image
    
    for key in new_keys:
        candidate_cams = key_to_prev_cams[key] # on présélectionne les caméras ayant vu le point 3D à reconstruire
        if len(candidate_cams) == 0:
            continue
    
        best_angle = -1
        best_U_world = None
    
        # On cherche la caméra passée qui donne le meilleur angle de parallaxe
        for cam_prev_idx in candidate_cams:
            C_prev = cams_centers[cam_prev_idx]
            P_prev = cams_P[cam_prev_idx]
    
            # Récupérer les coordonnées 2D du point dans les deux images
            # On utilise tracks_full pour trouver l'index du point 2D correspondant à la clé
            
            # Recherche de l'index 2D pour l'image actuelle
            mask_new = (tracks_full[new_im_id]['p3D_keys'] == key) #key : le point 3D qu'on cherche a reconstruire
            idx_in_tracks_new = np.where(mask_new)[0][0] 
            idx_2d_new = tracks_full[new_im_id]['p2D_ids'][idx_in_tracks_new]
            
            # Recherche de l'index 2D pour l'image précédente
            mask_prev = (tracks_full[cam_prev_idx]['p3D_keys'] == key)
            idx_in_tracks_prev = np.where(mask_prev)[0][0]
            idx_2d_prev = tracks_full[cam_prev_idx]['p2D_ids'][idx_in_tracks_prev]
            
            # une fois qu'on a les indexes on récupère les coordonnées :
            pt2d_new = p[new_im_id][idx_2d_new, :2]
            pt2d_prev = p[cam_prev_idx][idx_2d_prev, :2]
    
            # Triangulation temporaire (pour calculer l'angle entre les caméras)
            U_hom = cv.triangulatePoints(P_prev, P_new, pt2d_prev.T, pt2d_new.T)
            U_cand = (U_hom[:3] / U_hom[3]).flatten()
    
            # Calcul de l'angle de parallaxe
            v1 = U_cand - C_new
            v2 = U_cand - C_prev
            
            # Calcul de l'angle entre les caméras 
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))
    
            if angle > best_angle:
                best_angle = angle
                best_U_world = U_cand
    
        #  Validation par seuil d'angle
        if best_angle > min_parallax_angle:
            new_points_3D.append(best_U_world)
            keys_actually_added.append(key)
    
    
    # Mise à jour du modèle 3D avec les points reconstruits par triangulation
    if new_points_3D:
        n_Uw_before = Uw.shape[0]
        Uw = np.vstack((Uw, np.array(new_points_3D)))
        
        # Mettre à jour le mapping global
        for i, key in enumerate(keys_actually_added):
            p3D_keys_to_ids[key] = n_Uw_before + i
        
        p3D_keys_reconstructed = np.union1d(p3D_keys_reconstructed, keys_actually_added)
    
        # Mettre à jour les tracks de toutes les caméras
        for cam_idx in range(new_im_id + 1):  # Inclut les anciennes et la nouvelle
            # Trouver quelles clés parmi celles ajoutées sont vues par cette caméra
            keys_seen_by_cam, ids_in_added, ids_in_full = np.intersect1d(
                keys_actually_added, 
                tracks_full[cam_idx]['p3D_keys'], 
                return_indices=True
            )
            
            if len(keys_seen_by_cam) > 0:
                new_p2d_ids = tracks_full[cam_idx]['p2D_ids'][ids_in_full]
                
                if cam_idx < len(tracks):
                    # On ajoute les nouvelles clés aux tracks existants de l'ancienne caméra
                    tracks[cam_idx]['p3D_keys'] = np.append(tracks[cam_idx]['p3D_keys'], keys_seen_by_cam)
                    tracks[cam_idx]['p2D_ids'] = np.append(tracks[cam_idx]['p2D_ids'], new_p2d_ids)
                else:
                    # C'est la nouvelle caméra, on crée son premier track
                    tracks.append({
                        'p3D_keys': keys_seen_by_cam,
                        'p2D_ids': new_p2d_ids
                    })
    else:
        # Si aucun point n'est ajouté, pour garantir que lea taille de tracks reste égale au nb de caméras traitées.
        tracks.append({'p3D_keys': np.array([]), 'p2D_ids': np.array([])})
    
    print(f"Points orphelins : {len(new_keys)} | Triangulés avec succès : {len(new_points_3D)}")
        
    #%% Bundle adjustment
    
    BA_global = BA_LM_schur(
        Mwc, 
        Uw, 
        p3D_keys_to_ids, 
        tracks, 
        K, 
        p, 
    maxIt=20 # Reduced iterations for real-time feedback; increase for higher precision
    )

    BA_global.optimize()
    Mwc = BA_global.getPoses()
    Uw = BA_global.getPointCloud()
        
    Mwc, Uw = utils.normalizeReconstructionScale(Mwc,Uw)
    
    # Reprojection error after global BA
    global_ba_rmse = BA_global.compute_cost()
    print(f"[Image {new_im_id}] Global BA RMSE: {global_ba_rmse:.4f} pix")
    reproj_err_global_ba_list.append(global_ba_rmse)

print('Total time : {} sec'.format(time.time()-t0))
print('Localization - Mean reproj. error : {:.4f} pix'.format(np.mean(reproj_err_list)))
print('Localization - Max  reproj. error : {:.4f} pix'.format(np.max(reproj_err_list)))
print('Localization - Min  reproj. error : {:.4f} pix'.format(np.min(reproj_err_list)))
print('Global BA   - Mean RMSE : {:.4f} pix'.format(np.mean(reproj_err_global_ba_list)))
print('Global BA   - Max  RMSE : {:.4f} pix'.format(np.max(reproj_err_global_ba_list)))
print('Global BA   - Min  RMSE : {:.4f} pix'.format(np.min(reproj_err_global_ba_list)))

    
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
