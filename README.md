# MoCoLoR

Motion-compensated low-rank reconstruction for simultaneous structural and functional UTE lung MRI

![Figure1-methods](https://github.com/PulmonaryMRI/MoCoLoR/assets/22829346/8196dd2c-01df-49a3-8843-533f20f9cb41)

## Reference

Tan F, Zhu X, Chan M, Zapala MA, Vasanawala SS, Ong F, Lustig M, Larson PEZ. Motion-compensated low-rank reconstruction for simultaneous structural and functional UTE lung MRI. Magn Reson Med. 2023. doi: [10.1002/mrm.29703](https://dx.doi.org/10.1002/mrm.29703)

## Dependency

numpy>=1.21, matplotlib, sigpy==0.1.16, antspyx, h5py, pydicom

## Example Usage

```
# convert ute data when using the uwute sequence
# uses the output 'MRI_Raw.h5' of pcvipr_recon_binary created using -export_kdata flag
python convert_uwute_npy.py ${file_dir} ${file_dir}

# run xd reconstruction
python recon_xdgrasp_npy.py ${file_dir} --res_scale 1 --vent_flag 1

# run lor reconstruction
python recon_lrmoco_npy.py ${file_dir} --lambda_lr 0.05 --vent_flag 1 --mr_cflag 0

# run mocolor reconstruction
python recon_lrmoco_vent_npy.py ${file_dir} --lambda_lr 0.05 --vent_flag 1 --reg_flag 1

# rm temp files
rm bcoord.npy bdcf.npy bksp.npy _M_mr.npy

```
