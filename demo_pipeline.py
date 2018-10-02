#!/usr/bin/env python

"""
Complete demo pipeline for motion correction, source extraction, and
deconvolution of two-photon calcium imaging data using the CaImAn package.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev
"""

from __future__ import division
from __future__ import print_function
from builtins import range

import os
import sys
import cv2
import glob

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        print("Running under iPython")
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import time

import caiman as cm
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto

import argparse
import logging
from os.path import basename
#%%

#%%  handle command line arguments and activate logging

parser = argparse.ArgumentParser(description='Execute the CaImAn (Calcium Imaging Analysis) pipeline on a video recording (.tif file or .tiff stack). Motion correct, determine neuronal regions, extract signals and denoise.')
parser.add_argument('infile', type=str, nargs=1, help='file name (for .tif file) or folder (for .tiff stack)')
parser.add_argument('outfile',type=str, nargs=1, help='file name under which to store the output .npz file')
parser.add_argument('--log_fname', type=str, nargs='?', default="caiman_processing.log", help='file name under which to save a progress log. leave out to save to caiman_processing.log')
parser.add_argument('--mc_fname', type=str, nargs='?', default="", help='file name under which to save motion-corrected video. leave out to not save (default behavior)')
parser.add_argument('--nomc', action='store_true', help='if used, then no motion correction will be run on the input video')
parser.add_argument('--slurmid', type=int, nargs='?', default=0, help='slurm ID for multi-node processing. leave out to default to 0')
args = vars(parser.parse_args())

infile = args['infile']
outfile = args['outfile'][0]
log_fname = args['log_fname']
mc_fname = args['mc_fname']
nomc = args['nomc']
slurmid = args['slurmid']

splitlog = log_fname.split('.')
head = str.join('.',splitlog[:-1])
tail = '.' + splitlog[-1]
log_fname = head + '_' + str(slurmid) + tail 
logging.basicConfig(filename=log_fname,level=logging.INFO) 
#%%

def main():
    pass # For compatibility between running under Spyder and the CLI

#%% First setup some parameters

    # num processes
    n_proc = 12
    # dataset dependent parameters
    display_images = False              # Set this to true to show movies and plots
    fname = infile  # filename to be processed
    fr = 30                             # imaging rate in frames per second
    decay_time = 0.4                    # length of a typical transient in seconds

    # motion correction parameters
    niter_rig = 1               # number of iterations for rigid motion correction
    max_shifts = (6, 6)         # maximum allow rigid shift
    # for parallelization split the movies in  num_splits chuncks across time
    splits_rig = 56
    # start a new patch for pw-rigid motion correction every x pixels
    strides = (48, 48)
    # overlap between pathes (size of patch strides+overlaps)
    overlaps = (24, 24)
    # for parallelization split the movies in  num_splits chuncks across time
    splits_els = 56
    upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    # parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thresh = 0.8          # merging threshold, max correlation allowed
    # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    rf = 15
    stride_cnmf = 6             # amount of overlap between the patches in pixels
    K = 4                       # number of components per patch
    gSig = [4, 4]               # expected half size of neurons
    # initialization method (if analyzing dendritic data using 'sparse_nmf')
    init_method = 'greedy_roi'
    is_dendrites = False        # flag for analyzing dendritic data
    # sparsity penalty for dendritic data analysis through sparse NMF
    alpha_snmf = None

    # parameters for component evaluation
    min_SNR = 2.5               # signal to noise ratio for accepting a component
    rval_thr = 0.8              # space correlation threshold for accepting a component
    cnn_thr = 0.8               # threshold for CNN based classifier

#%% download the dataset if it's not present in your folder
    #if fname[0] in ['Sue_2x_3000_40_-46.tif', 'demoMovie.tif']:
    #   fname = [download_demo(fname[0])]


    print('Loading movie');

#%% play the movie
    # playing the movie using opencv. It requires loading the movie in memory. To
    # close the video press q

    m_orig = cm.load_movie_chain(fname[:1])
    downsample_ratio = 0.2
    offset_mov = -np.min(m_orig[:100])
    moviehandle = m_orig.resize(1, 1, downsample_ratio)
    if display_images:
        moviehandle.play(gain=10, offset=offset_mov, fr=30, magnification=2)

    print('Movie loaded. Starting cluster')

    
#%% start a cluster for parallel processing
    t_cs = time.time()
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_proc, single_thread=False)
    
    t_cf = time.time()
    print('Cluster started in ', int(t_cf - t_cs), ' seconds')
    time.sleep(4)
    print('Beginning motion correction')

#%%% MOTION CORRECTION

    t_ms = time.time()
    # first we create a motion correction object with the parameters specified
    min_mov = cm.load(fname[0], subindices=range(200)).min()
    # this will be subtracted from the movie to make it non-negative
    if not nomc:
      mc = MotionCorrect(fname[0], min_mov,
          dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,splits_rig=splits_rig,strides=strides, overlaps=overlaps, splits_els=splits_els,upsample_factor_grid=upsample_factor_grid,max_deviation_rigid=max_deviation_rigid,shifts_opencv=True, nonneg_movie=True)
		# note that the file is not loaded in memory

		#%% Run piecewise-rigid motion correction using NoRMCorre
      mc.motion_correct_pwrigid(save_movie=True)
      m_els = cm.load(mc.fname_tot_els)
      bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
      t_mf = time.time()
      print('Motion correction complete in ', int(t_mf - t_ms),' seconds')
		# maximum shift to be used for trimming against NaNs
		#%% compare with original movie
      moviehandle = cm.concatenate([m_orig.resize(1, 1, downsample_ratio) + offset_mov, m_els.resize(1, 1, downsample_ratio)],axis=2)
    
      display_images = False
      if display_images:
        moviehandle.play(fr=60, q_max=99.5, magnification=2, offset=0)  # press q to exit

#%% MEMORY MAPPING
    # memory map the file in order 'C'
      fnames = mc.fname_tot_els   # name of the pw-rigidly corrected file.
      border_to_0 = bord_px_els     # number of pixels to exclude
      fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
			       border_to_0=bord_px_els)  # exclude borders

    display_images = False
    fname_new = 'memmap__d1_512_d2_512_d3_1_order_C_frames_3024_.mmap'
    bord_px_els = 5
    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

#%% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=n_proc, single_thread=False)

#%% RUN CNMF ON PATCHES

    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)
    t1 = time.time()

    cnm = cnmf.CNMF(n_processes=n_proc, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf,
                    only_init_patch=False, gnb=gnb, border_pix=bord_px_els)
    cnm = cnm.fit(images)
    
    t2 = time.time()
    print('Initial CNMF fit complete in ', int(t2-t1), 'seconds.')
#%% plot contours of found components
    Cn = cm.local_correlations(images.transpose(1, 2, 0))
    Cn[np.isnan(Cn)] = 0
    #plt.figure()
    #crd = plot_contours(cnm.A, Cn, thr=0.9)
    #plt.title('Contour plots of found components')


#%% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier
    t3 = time.time()
    print('estimating component quality')
    
    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
        estimate_components_quality_auto(images, cnm.A, cnm.C, cnm.b, cnm.f,
                                         cnm.YrA, fr, decay_time, gSig, dims,
                                         dview=dview, min_SNR=min_SNR,
                                         r_values_min=rval_thr, use_cnn=False,
                                         thresh_cnn_min=cnn_thr)
    t4 = time.time()
    print('Component quality estimation complete in ', int(t4-t3),' seconds')
#%% PLOT COMPONENTS

    if display_images:
        plt.figure()
        plt.subplot(121)
        crd_good = cm.utils.visualization.plot_contours(
            cnm.A[:, idx_components], Cn, thr=.8, vmax=0.75)
        plt.title('Contour plots of accepted components')
        plt.subplot(122)
        crd_bad = cm.utils.visualization.plot_contours(
            cnm.A[:, idx_components_bad], Cn, thr=.8, vmax=0.75)
        plt.title('Contour plots of rejected components')

#%% VIEW TRACES (accepted and rejected)

    if display_images:
        view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components], cnm.C[idx_components],
                         cnm.b, cnm.f, dims[0], dims[1], YrA=cnm.YrA[idx_components],
                         img=Cn)

        view_patches_bar(Yr, cnm.A.tocsc()[:, idx_components_bad], cnm.C[idx_components_bad],
                         cnm.b, cnm.f, dims[0], dims[1], YrA=cnm.YrA[idx_components_bad],
                         img=Cn)

#%% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    print('re-running cnmf')
    t5 = time.time()
    A_in, C_in, b_in, f_in = cnm.A[:,
                                   idx_components], cnm.C[idx_components], cnm.b, cnm.f
    cnm2 = cnmf.CNMF(n_processes=n_proc, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                     merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                     f_in=f_in, rf=None, stride=None, gnb=gnb,
                     method_deconvolution='oasis', check_nan=True)

    cnm2 = cnm2.fit(images)
    t6 = time.time()
    print('CNMF re-run complete in ', int(t6-t5), ' seconds')
#%% Extract DF/F values

    F_dff = detrend_df_f(cnm2.A, cnm2.b, cnm2.C, cnm2.f, YrA=cnm2.YrA,
                         quantileMin=8, frames_window=250)

#%% Show final traces
    #cnm2.view_patches(Yr, dims=dims, img=Cn)

    save_results = True
    if save_results:
        np.savez_compressed(outfile,Cn=Cn, A=cnm2.A.todense(), C=cnm2.C,b=cnm2.b, f=cnm2.f, YrA=cnm2.YrA, d1=d1, d2=d2,idx_components=idx_components, idx_components_bad=idx_components_bad)

    
#%% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
   

 
#%% reconstruct denoised movie
    denoised = cm.movie(cnm2.A.dot(cnm2.C) +
                        cnm2.b.dot(cnm2.f)).reshape(dims + (-1,), order='F').transpose([2, 0, 1])

#%% play along side original data
    #moviehandle = cm.concatenate([m_els.resize(1, 1, downsample_ratio),
     #               denoised.resize(1, 1, downsample_ratio)],
      #             axis=2)
    if display_images:
            moviehandle.play(fr=60, gain=15, magnification=2, offset=0)  # press q to exit

#%%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
