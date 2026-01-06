import numpy as np
from BA_LM_schur import BA_LM_schur

def BA_LM_localization(Mwc_init, Xw, x2d, K, maxIt=20):

    tracks = [{
        'p3D_keys': np.arange(Xw.shape[0]),
        'p2D_ids': np.arange(x2d.shape[0])
    }]

    p = [x2d]
    p3D_keys_to_ids = np.arange(Xw.shape[0])
    Mwc_init_list = [Mwc_init]

    BA = BA_LM_schur(
        Mwc_init=Mwc_init_list,
        Uw_init=Xw.copy(),
        p3D_keys_to_ids=p3D_keys_to_ids,
        tracks=tracks,
        K=K,
        p=p,
        maxIt=maxIt
    )
    
    Uw_fixed = BA.Uw.copy()
    BA.optimize()
    BA.Uw[:] = Uw_fixed

    return BA.getPoses()[0]