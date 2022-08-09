# MoCoLoR

## Dependency

numpy, matplotlib, sigpy==0.1.16, antspyx, h5py, pydicom

## Example Usage

```
# convert ute
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
