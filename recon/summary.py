# %% Results
import sigpy.plot as pl
import numpy as np

# Usage
# python recon_lrmoco_npy.py data/floret/ --lambda_lr 0.05 --vent_flag 1 --mr_cflag 0  --res_scale 1.5 --vent_flag 1 --fov_x 128 --fov_y 128 --fov_z 128

folder = "data/floret/"
# folder = "data/floret-la/"

# XD GRASP
try:
    x = np.load(folder + "sv_xdgrasp.npy")
    print(x.shape)
    pl.ImagePlot(x, x=2,y=3,z=0)

    x = np.load(folder + "jac_xdgrasp.npy")
    print(x.shape)
    pl.ImagePlot(x, x=2,y=3,z=0)
    
    x = np.load(folder + "prL.npy")
    x = np.rot90(x, k=2, axes=(2,3))
    print(x.shape)
    pl.ImagePlot(x[:,:, x.shape[2]//2,:], x=1,y=2,z=0, hide_axes=True)
except:
    print("Could not locate xd-grasp data.")

# LOR
try:
    x = np.load(folder + "sv_lor.npy")
    print(x.shape)
    pl.ImagePlot(x, x=2,y=3,z=0)

    x = np.load(folder + "jac_lor.npy")
    print(x.shape)
    pl.ImagePlot(x, x=2,y=3,z=0)
    
    x = np.load(folder + "lor_vent.npy")
    x = np.rot90(x, k=2, axes=(2,3))
    print(x.shape)
    pl.ImagePlot(x[:,:, x.shape[2]//2,:], x=1,y=2,z=0, hide_axes=True)
except:
    print("Could not locate LoR data.")

# MoCoLoR
try:
    x = np.load(folder + "_M_mr.npy")
    print(x.shape)
    pl.ImagePlot(x, x=2,y=3,z=0)

    x = np.load(folder + "sv_mocolor_vent.npy")
    print(x.shape)
    pl.ImagePlot(x, x=2,y=3,z=0)

    x = np.load(folder + "jac_mocolor_vent.npy")
    print(x.shape)
    pl.ImagePlot(x, x=2,y=3,z=0)

    x = np.load(folder + "mocolor_vent.npy")
    x = np.rot90(x, k=2, axes=(2,3))
    print(x.shape)
    pl.ImagePlot(x[:,:, x.shape[2]//2,:], x=1,y=2,z=0, hide_axes=True)
except:
    print("Could not locate MoCoLoR data.")
