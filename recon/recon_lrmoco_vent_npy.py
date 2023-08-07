#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:31:08 2022

@author: ftan1
"""

import argparse 

import sigpy as sp
import scipy.ndimage as ndimage_c
import numpy as np

import sys
# sys.path.append("./sigpy_mc/")
import sigpy_e.cfl as cfl 
import sigpy_e.ext as ext

import sigpy_e.prox as prox
import sigpy_e.reg as reg
from sigpy_e.linop_e import NFTs,Diags,DLD,Vstacks
import sigpy.mri as mr
import os

if __name__ == '__main__':


    ## IO parameters
    parser = argparse.ArgumentParser(description='motion compensated low rank constrained recon.')

    parser.add_argument('--res_scale', type=float, default=1,
                        help='scale of resolution, full res == 1')
    parser.add_argument('--fov_x', type=float, default=1,
                        help='scale of FOV x, full res == 1')
    parser.add_argument('--fov_y', type=float, default=1,
                        help='scale of FOV y, full res == 1')
    parser.add_argument('--fov_z', type=float, default=1,
                        help='scale of FOV z, full res == 1')

    parser.add_argument('--n_ref', type=int, default=0,
                        help='reference frame')
    parser.add_argument('--reg_flag', type=int, default=1,
                        help='derive motion field from registration')
    parser.add_argument('--vent_flag', type=int, default=0,
                        help='output jacobian determinant and specific ventilation')
    parser.add_argument('--mr_cflag', type=int, default=1,
                        help='Resp motion compensation')
    
    parser.add_argument('--lambda_lr', type=float, default=1e-2,
                        help='low rank regularization, default is 0.01')
    parser.add_argument('--iner_iter', type=int, default=5,
                        help='Num of inner iterations.')
    parser.add_argument('--outer_iter', type=int, default=3,
                        help='Num of outer iterations.')
    parser.add_argument('--sup_iter', type=int, default=3,
                        help='Num of superior iterations.')
    
    parser.add_argument('--device', type=int, default=0,
                        help='Computing device.')
    parser.add_argument('fname', type=str,
                        help='Prefix of raw data and output(_mocolor).')
    # a set of CFL files, including(kspace, trajectory, and density_compensation_weighting)
    args = parser.parse_args()


    #
    res_scale = args.res_scale
    fname = args.fname
    lambda_lr = args.lambda_lr
    device = args.device
    outer_iter = args.outer_iter
    iner_iter = args.iner_iter
    sup_iter = args.sup_iter
    fov_scale = (args.fov_x, args.fov_y, args.fov_z)
    n_ref = args.n_ref
    reg_flag = args.reg_flag
    mr_cflag = args.mr_cflag
    vent_flag = args.vent_flag
    
    ## data loading
    data = np.load(os.path.join(fname, 'bksp.npy'))
    traj = np.real(np.load(os.path.join(fname, 'bcoord.npy')))
    dcf = np.sqrt(np.load(os.path.join(fname, 'bdcf.npy')))
    
    nf_scale = res_scale
    nf_arr = np.sqrt(np.sum(traj[0,0,:,:]**2,axis = 1)) 
    nf_e = np.sum(nf_arr<np.max(nf_arr)*nf_scale)
    scale = fov_scale
    traj[...,0] = traj[...,0]*scale[0]
    traj[...,1] = traj[...,1]*scale[1]
    traj[...,2] = traj[...,2]*scale[2]

    traj = traj[...,:nf_e,:]
    data = data[...,:nf_e]
    dcf = dcf[...,:nf_e]

    nphase,nCoil,npe,nfe = data.shape
    tshape = (int(np.max(traj[...,0])-np.min(traj[...,0]))
              ,int(np.max(traj[...,1])-np.min(traj[...,1]))
              ,int(np.max(traj[...,2])-np.min(traj[...,2])))

    ## calibration
    print('Calibration...')
    ksp = np.reshape(np.transpose(data,(1,0,2,3)),(nCoil,nphase*npe,nfe))
    dcf2 = np.reshape(dcf**2,(nphase*npe,nfe))
    coord = np.reshape(traj,(nphase*npe,nfe,3))

    mps = ext.jsens_calib(ksp,coord,dcf2,device = sp.Device(device),ishape = tshape)
    S = sp.linop.Multiply(tshape, mps)

    ## registration
    print('Motion Field Initialization...')
    # M_fields = []
    # iM_fields = []
    # if reg_flag == 1:
    #     imgL = np.load(os.path.join(fname, 'prL.npy'))
    #     imgL = np.abs(np.squeeze(imgL))
    #     imgL = imgL/np.max(imgL)
    #     for i in range(nphase):
    #         M_field, iM_field = reg.ANTsReg(imgL[n_ref], imgL[i])
    #         M_fields.append(M_field)
    #         iM_fields.append(iM_field)
    #     M_fields = np.asarray(M_fields)
    #     iM_fields = np.asarray(iM_fields)
    #     np.save(os.path.join(fname, '_M_mr.npy'),M_fields)
    #     np.save(os.path.join(fname, '_iM_mr.npy'),iM_fields)
    # else:
    #     M_fields = np.load(os.path.join(fname, '_M_mr.npy'))
    #     iM_fields = np.load(os.path.join(fname, '_iM_mr.npy'))

    # iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
    # M_fields = [M_fields[i] for i in range(M_fields.shape[0])]
    
    # ######## TODO scale M_field
    # print('Motion Field scaling...')
    # M_fields = [reg.M_scale(M,tshape) for M in M_fields]
    # iM_fields = [reg.M_scale(M,tshape) for M in iM_fields]
    
    M_fields = np.zeros((nphase,) + tshape + (len(tshape),))
    
    ## low rank
    print('Low rank prep...')
    PFTSs = []
    for i in range(nphase):
        FTs = NFTs((nCoil,)+tshape,traj[i,...],device=sp.Device(device))
        W = sp.linop.Multiply((nCoil,npe,nfe,),dcf[i,:,:]) 
        FTSs = W*FTs*S
        PFTSs.append(FTSs)
    PFTSs = Diags(PFTSs,oshape=(nphase,nCoil,npe,nfe,),ishape=(nphase,)+tshape)

    if mr_cflag == 1:
        print('With moco...')
        sp.linop.Identity((nphase,)+tshape)
        Ms = []
        # M0s = []
        for i in range(nphase):
            # M = reg.interp_op(tshape,iM_fields[i],M_fields[i])
            M = reg.interp_op(tshape,M_fields[i])
            # M0 = reg.interp_op(tshape,np.zeros(tshape+(3,)))
            M = DLD(M,device=sp.Device(device))
            # M0 = DLD(M0,device=sp.Device(device))
            Ms.append(M)
            # M0s.append(M0)
        Ms = Diags(Ms,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)
        # M0s = Diags(M0s,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)
        # LRM = GLRA((nphase,)+tshape,lambda_lr,A=Ms)
    else:
        print('Without moco...')
        Ms = sp.linop.Identity((nphase,)+tshape)
        # M0s = sp.linop.Identity((nphase,)+tshape)
    LR = prox.GLRA((nphase,)+tshape,lambda_lr)

    ## precondition
    print('Preconditioner calculation...')
    tmp = FTSs.H*FTSs*np.complex64(np.ones(tshape))
    L=np.mean(np.abs(tmp))
    # TODO condition number calc
    tmp = np.zeros(tshape)
    tmp[0,0,0]=1.0
    tmp = np.fft.fftshift(tmp)
    tmp = FTSs.H*FTSs*np.complex64(tmp)
    tmp = np.fft.ifftshift(tmp)
    # TODO condition number calc
    wdata = data*dcf[:,np.newaxis,:,:]

    # ADMM
    print('Recon...')
    # Ms is merge
    qt = np.zeros((nphase,)+tshape,dtype=np.complex64)
    u0 = np.zeros_like(qt)
    z0 = np.zeros_like(qt)

    rho = 1
    #ATA = lambda x : 1/L*PFTSs.H*PFTSs*x + Ms.H*Ms*x
    b0 = 1/L*PFTSs.H*wdata
    res_list = []
    
    for k in range(sup_iter):
        for i in range(outer_iter):
            b = b0 + rho*Ms.H*(z0 - u0)
            # CG_step = sp.alg.ConjugateGradient(ATA, b, qt, max_iter=iner_iter,tol=1e-7)
            # grad = lambda x : 1/L*PFTSs.H*PFTSs*x + rho*Ms.H*Ms*x - b
            grad = lambda x : 1/L*PFTSs.H*PFTSs*x + rho*x - b
            GD_step = sp.alg.GradientMethod(grad,qt,.1,accelerate=False,tol=5e-7)
            for j in range(iner_iter):
                # CG_step.update()
                GD_step.update()
                # qt = qt - 0.2*(1/L*PFTSs.H*(PFTSs*qt - wdata) + Ms.H*(Ms*qt - z0 + u0))
                res_norm = GD_step.resid/np.linalg.norm(qt)*GD_step.alpha
                print('superior iter:{}, outer iter:{}, inner iter:{},res:{}'.format(k,i,j,res_norm))
                if res_norm <5e-8:
                    break
                res_list.append(res_norm)
            z0 = np.complex64(LR(1,Ms*qt + u0))
            u0 = u0 + (Ms*qt - z0)
            
        # update motion field
        #print('Registration...')
        M_fields = []
        # iM_fields = []
        if reg_flag == 1:
            imgL = qt
            imgL = np.abs(np.squeeze(imgL))
            imgL = imgL/np.max(imgL)
            for i in range(nphase):
                M_field, iM_field = reg.ANTsReg(imgL[n_ref], imgL[i])
                M_fields.append(M_field)
                # iM_fields.append(iM_field)
            M_fields = np.asarray(M_fields)
            # iM_fields = np.asarray(iM_fields)
            np.save(os.path.join(fname, '_M_mr.npy'),M_fields)
            # np.save(os.path.join(fname, '_iM_mr.npy'),iM_fields)
        else:
            M_fields = np.load(os.path.join(fname, '_M_mr.npy'))
            # iM_fields = np.load(os.path.join(fname, '_iM_mr.npy'))
    
        # iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
        M_fields = [M_fields[i] for i in range(M_fields.shape[0])]
    
        ######## TODO scale M_field to np array
        #print('Motion Field scaling...')
        M_fields = [reg.M_scale(M,tshape) for M in M_fields]
        # iM_fields = [reg.M_scale(M,tshape) for M in iM_fields]
        
        
        #print('With moco...')
        sp.linop.Identity((nphase,)+tshape)
        Ms = []
        # M0s = []
        for i in range(nphase):
            # M = reg.interp_op(tshape,iM_fields[i],M_fields[i])
            M = reg.interp_op(tshape,M_fields[i])
            # M0 = reg.interp_op(tshape,np.zeros(tshape+(3,)))
            M = DLD(M,device=sp.Device(device))
            # M0 = DLD(M0,device=sp.Device(device))
            Ms.append(M)
            # M0s.append(M0)
        Ms = Diags(Ms,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)
        # M0s = Diags(M0s,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)

        np.save(os.path.join(fname, 'mocolor_vent.npy'), qt)
        np.save(os.path.join(fname, 'mocolor_vent_residual.npy'), np.asarray(res_list))
        
    # qt = np.load(os.path.join(fname, 'mocolor_vent.npy'))
    # nphase = 6
    # jacobian determinant & specific ventilation
    if vent_flag==1:
        print('Jacobian Determinant and Specific Ventilation...')
        jacs = []
        svs = []
        qt = np.abs(np.squeeze(qt))
        qt = qt/np.max(qt)
        for i in range(nphase):
            jac, sv = reg.ANTsJac(np.abs(qt[n_ref]), np.abs(qt[i]))
            jacs.append(jac)
            svs.append(sv)
        jacs = np.asarray(jacs)
        svs = np.asarray(svs)
        np.save(os.path.join(fname, 'jac_mocolor_vent.npy'), jacs)
        np.save(os.path.join(fname, 'sv_mocolor_vent.npy'), svs)