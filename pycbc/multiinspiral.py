#!/usr/bin/env python

# Copyright (C) 2014 Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import logging
from collections import defaultdict
import argparse
from pycbc import vetoes, psd, waveform, events, strain, scheme, fft,\
    DYN_RANGE_FAC
from pycbc.filter import MatchedFilterControl
from pycbc.types import TimeSeries, zeros, float32, complex64
from pycbc.types import MultiDetOptionAction
import pycbc.detector
import pycbc.weave
import numpy as np
import healpy as hp
from itertools import combinations

import time

def network_chisq_grid(chisq_unique,chisq_dof_unique,coinc_idx_unique,coinc_idx_det_frame,coinc_idx_list,coherent_ifo_triggers):
    """
    Input: chisq_unique: array of individual det chisq values for unique triggers
           chisq_dof: array of individual det chisq degrees of freedom values for unique triggers
           coind_idx_unique: array of unique sets of triggers with 3 columns for: detector frame trigger id, coherent trigger id, SNR  triggers for these events ids
           coinc_idx_det_frame : list of dictionaries with detector frame ids for each sky location/timeslide
           coinc_idx_list : list of arrays of coherent trigger ids for each sky location/timeslide
           coherent_ifo_triggers : list of dictionaries with detector SNR s for each trigger.
    Output: network_chisq_list : network chisqs for each set of triggers for each sky location/timeslide
           chisq_list : list of dictionaires with detector chisq for triggers at each sky location/timeslide
           chisq_dof_list : list of dictionaires with detector chisq  dof for triggers at each sky location/timeslide
    """
    network_chisq_list=[]
    chisq_list=[]
    chisq_dof_list=[]
    for i, indices in enumerate(coinc_idx_det_frame):
        chisq={}
        chisq_dof={}
        for ifo in indices.keys():
            unique_coinc_id= np.array(list(map(int,coinc_idx_unique[ifo][:,1])))
            unique_det_id= np.array(list(map(int,coinc_idx_unique[ifo][:,0])))
            trig_mask=np.array([np.intersect1d(np.where(unique_coinc_id==coinc_idx),np.where(unique_det_id==det_idx))[0] for coinc_idx, det_idx in zip(coinc_idx_list[i],indices[ifo])])
            if len(trig_mask)>0:
                chisq[ifo]=chisq_unique[ifo][trig_mask] 
                chisq_dof[ifo]=chisq_dof_unique[ifo][trig_mask]
            else:
                chisq[ifo]=np.array([])
                chisq_dof[ifo]=np.array([])
        network_chisq_list.append(network_chisq(chisq,chisq_dof,coherent_ifo_triggers[i]))
        chisq_list.append(chisq)
        chisq_dof_list.append(chisq_dof)
    return network_chisq_list,chisq_list,chisq_dof_list

def network_chisq(chisq, chisq_dof, snr_dict):
    """
    Input : chisq : dictionary of detector chisq for each detector
            chisq_dof : dictionary with degrees of freedom for the chisq for each detector
            snr_dict : dictionary snr triggers for each detector
    Output : network_chisq : coherent chisqs combining chisq of each detector for each event.
    """
    ifos = sorted(snr_dict.keys())
    chisq_per_dof = {}
    for ifo in ifos:
        chisq_per_dof[ifo] = chisq[ifo] / chisq_dof[ifo]
        chisq_per_dof[ifo][chisq_per_dof[ifo]<1] = 1
    snr2 = {ifo : np.real(np.array(snr_dict[ifo]) *
                np.array(snr_dict[ifo]).conj()) for ifo in ifos}
    coinc_snr2 = sum(snr2.values())
    snr2_ratio = {ifo : snr2[ifo] / coinc_snr2 for ifo in ifos}
    network_chisq = sum( [chisq_per_dof[ifo] * snr2_ratio[ifo] for ifo in ifos] )
    return network_chisq

def pycbc_reweight_snr_grid(rho_coh_list, network_chisq_list, a = 3, b = 1. / 6.):
    """
    Input:  rho_coh_list:  list of Dictoinaries of coincident or coherent SNR at different sky locations/timeslides
            network_chisq_list: a list of sets of chisq values for each trigger for each sky location/ timeslide 
    Output: reweighted_snr_list: list  of Reweighted SNRs for each sky location/timeslide
    """
    reweighted_snr_list=[]
    for rho_coh, network_chisq in zip(rho_coh_list,network_chisq_list):
        reweighted_snr_list.append(pycbc_reweight_snr(rho_coh,network_chisq))
    return reweighted_snr_list

def pycbc_reweight_snr(network_snr, network_chisq, a = 3, b = 1. / 6.):
    """
    Input:  network_snr:  Dictoinary of coincident or coherent SNR for each
                          trigger
            network_chisq: A chisq value for each trigger 
    Output: reweighted_snr: Reweighted SNR for each trigger
    """
    denom = ((1 + network_chisq)**a) / 2
    reweighted_snr = network_snr / denom**b
    return reweighted_snr

def reweight_snr_by_null(network_snr, nullsnr):
    """
    Input:  network_snr:  Dictoinary of coincident, coherent, or reweighted
                          SNR for each trigger
            null: Null snr for each trigger
    Output: reweighted_snr: Reweighted SNR for each trigger
    """
    nullsnr[nullsnr <= 4.25] = 4.25
    reweighted_snr = network_snr / (nullsnr - 3.25)
    return reweighted_snr

def get_weighted_antenna_patterns(Fp_dict, Fc_dict, sigma_dict):
    """
    Input:  Fp_dict: Dictionary of the antenna response fuctions to plus
                     polarisation for each ifo
            Fc_dict: Dictionary of the antenna response fuctions to cross
                     polarisation for each ifo
           sigma_dict: Sigma dictionary for each ifo (sensitivity of each ifo)
    Output: wp: 1 x nifo of the weighted antenna response fuctions to plus
                polarisation for each ifo
            wc: 1 x nifo of the weighted antenna response fuctions to cross
                polarisation for each ifo
    """
    #Need the keys to be in alphabetical order
    keys = sorted(sigma_dict.keys())
    wp = np.array([sigma_dict[ifo]*Fp_dict[ifo] for ifo in keys])
    wc = np.array([sigma_dict[ifo]*Fc_dict[ifo] for ifo in keys])
    if not isinstance(Fp_dict[list(keys)[0]], float):
        wp=wp.T
        wc=wc.T
    return wp, wc

def get_projection_matrix_grid(wp, wc):
    """
    Input:  wp,wc: The weighted antenna response fuctions to plus and cross
                   polarisations respectively
    Output: projection_matrix_list: list of matrices of the projection of  the data onto the signal space
    """
    if len(wp.shape)>1:
        proj_matrix_list = np.array([get_projection_matrix(wP,wC) for wP, wC in zip(wp,wc)])
    else:
        proj_matrix_list = np.array([get_projection_matrix(wp,wc)])
    return proj_matrix_list

def get_projection_matrix(wp, wc):
    """
    Input:  wp,wc: The weighted antenna response fuctions to plus and cross
                   polarisations respectively
    Output: projection_matrix: Projects the data onto the signal space
    """
    denominator = np.dot(wp, wp) * np.dot(wc, wc) - np.dot(wp, wc)**2
    projection_matrix = (np.dot(wc, wc)*np.outer(wp, wp) +
                         np.dot(wp, wp)*np.outer(wc, wc) -
                         np.dot(wp, wc)*(np.outer(wp, wc) +
                         np.outer(wc, wp))) / denominator
    return projection_matrix

def coherent_snr_grid(coinc_triggers_list, coinc_idx_list, threshold, projection_matrix_list,
                rho_coinc_list=[]):
    """
    Inputs: coinc_triggers_list : a list of dictionaries of the normalised complex snr time
                          series for each ifo for . The keys are the ifos (e.g.
                          'L1','H1', and 'V1')
            coinc_idx_list : A list of arrays of the indexes you want to analyse for each sky location/timeslide
            threshold : Triggers with rho_coh<threshold are cut
            projection_matrix_list : list of matrices produced by get_projection_matrix.
            rho_coinc_list : Optional- The coincident snr for each trigger.
    Output: rho_coh_list: a list of arrays of the coherent snr for the detector network
            index_list  : list of arrays of indices that survive cuts
            snrv_list   : list of dictionaries of individual ifo triggers that survive cuts
            coinc_snr_list: list of the coincident snr values for triggers surviving the
                       coherent cut
    """
    rho_coh_list=[]
    index_list=[]
    snrv_list=[]
    coinc_snr_list=[]
    for snr_triggers, index, projection_matrix, coinc_snr in zip(coinc_triggers_list, coinc_idx_list, projection_matrix_list, rho_coinc_list):
        rho_coh,index,snrv,coinc_snr=coherent_snr(snr_triggers,index,threshold,projection_matrix,coinc_snr)
        rho_coh_list.append(rho_coh)
        index_list.append(index)
        snrv_list.append(snrv)
        coinc_snr_list.append(coinc_snr)
    return rho_coh_list,index_list,snrv_list,coinc_snr_list

def coherent_snr(snr_triggers, index, threshold, projection_matrix,
                coinc_snr=[]):
    """
    Inputs: snr_triggers: is a dictionary of the normalised complex snr time
                          series for each ifo. The keys are the ifos (e.g.
                          'L1','H1', and 'V1')
            index  : An array of the indexes you want to analyse. Not used for
                     calculations, just for book keeping
            threshold: Triggers with rho_coh<threshold are cut
            projection_matrix: Produced by get_projection_matrix.
            coinc_snr: Optional- The coincident snr for each trigger.
    Output: rho_coh: an array of the coherent snr for the detector network
            index  : Indexes that survive cuts
            snrv   : Dictionary of individual ifo triggers that survive cuts
            coinc_snr: The coincident snr value for triggers surviving the
                       coherent cut
    """
    #Calculate rho_coh
    snr_array = np.array([snr_triggers[ifo]
                         for ifo in sorted(snr_triggers.keys())])
    x = np.inner(snr_array.conj().transpose(),projection_matrix)
    rho_coh2 = sum(x.transpose()*snr_array)
    rho_coh = np.sqrt(rho_coh2)
    #Apply thresholds
    index = index[rho_coh > threshold]
    if len(coinc_snr) != 0: coinc_snr = coinc_snr[rho_coh > threshold]
    snrv = {ifo : snr_triggers[ifo][rho_coh > threshold]
           for ifo in snr_triggers.keys()}
    rho_coh = rho_coh[rho_coh > threshold]
    return rho_coh, index, snrv, coinc_snr

def coincident_snr_grid(snr_dict, coinc_idx_list, time_delays):
    """
    Input: snr_dict: Dictionary of individual detector SNR
           coinc_idx_list   : list of arrays of geocent indices you want to find coinc SNR for
       time_delays : list of time delays to aply to SNR to find coincidences. 
    Output: rho_coinc_list: Coincident snr triggers
                            indexes that survive cuts
    """
    rho_coinc_list=[]
    if isinstance(time_delays[list(time_delays.keys())[0]],int):
        for ifo in snr_dict.keys():
            snr_dict[ifo].roll(-time_delays[ifo])
        #Restrict the snr timeseries to just the interesting points
        rho_coinc_list.append(coincident_snr(snr_dict, coinc_idx_list[0]))
        for ifo in snr_dict.keys():
            snr_dict[ifo].roll(time_delays[ifo])
    else:
        for i in range(len(time_delays[list(time_delays.keys())[0]])):
            for ifo in snr_dict.keys():
                snr_dict[ifo].roll(-time_delays[ifo][i])
            #Restrict the snr timeseries to just the interesting points
            rho_coinc_list.append(coincident_snr(snr_dict, coinc_idx_list[i]))
            for ifo in snr_dict.keys():
                snr_dict[ifo].roll(time_delays[ifo][i])
    return rho_coinc_list

def coincident_snr(snr_dict, index):
    """
    Input: snr_dict: Dictionary of individual detector SNRs in geocent time
           index   : Geocent indexes you want to find coinc SNR for
    Output: rho_coinc: Coincident snr triggers
                            indexes that survive cuts
    """
    #Restrict the snr timeseries to just the interesting points
    coinc_triggers = {ifo : snr_dict[ifo][index] for ifo in snr_dict.keys()}
    #Calculate the coincident snr
    snr_array = np.array([coinc_triggers[ifo]
                        for ifo in coinc_triggers.keys()])
    rho_coinc = np.sqrt(np.sum(snr_array * snr_array.conj(),axis=0))
    return rho_coinc

def coincident_snr_cut_grid(snr, coinc_idx_list, time_delays, rho_coinc_list, threshold):
    """
    Input: snr_dict: Dictionary of individual detector SNRs in geocent time
           coinc_idx_list   : indice lists you want to find coinc SNR for
           rho_coinc_list : list of coincidence SNR values for different sky locations/timeslides
           threshold: Indexes with coinc SNR below this threshold are cut
    Output: rho_coinc: Coincident snr triggers that pass cut
            index    : The subset of input index that survive the cuts
            coinc_triggers: Dictionary of individual detector SNRs at
                            indexes that survive cuts
    """
    rho_coinc_list_filt=[]
    coinc_idx_list_filt=[]
    coinc_triggers_list=[]
    for i,(index,rho_coinc) in enumerate(zip(coinc_idx_list,rho_coinc_list)):
        for ifo in snr.keys():
            snr[ifo].roll(-time_delays[ifo][i])
        rho_coinc_filt,index_filt,coinc_triggers = coincident_snr_cut(snr,index,rho_coinc,threshold)
        rho_coinc_list_filt.append(rho_coinc_filt)
        coinc_idx_list_filt.append(index_filt)
        coinc_triggers_list.append(coinc_triggers)
        for ifo in snr.keys():
            snr[ifo].roll(time_delays[ifo][i])
    return rho_coinc_list_filt, coinc_idx_list_filt, coinc_triggers_list


def coincident_snr_cut(snr_dict, index, rho_coinc, threshold):
    """
    Input: snr_dict: Dictionary of individual detector SNRs in geocent time
           index   : Geocent indexes you want to find coinc SNR for
           rho_coinc: Coincident SNR triggers
           threshold: Indexes with coinc SNR below this threshold are cut
    Output: rho_coinc: Coincident snr triggers that survive cut. 
            index    : The subset of input index that survive the cuts
            coinc_triggers: Dictionary of individual detector SNRs at
                            indexes that survive cuts
    """
    #Apply threshold
    thresh_indexes = rho_coinc > threshold
    index = index[thresh_indexes]
    coinc_triggers = {ifo : snr_dict[ifo][index] for ifo in snr_dict.keys()}
    rho_coinc = rho_coinc[thresh_indexes]
    return rho_coinc, index, coinc_triggers


def null_snr_grid(rho_coh_list, rho_coinc_list, null_min=5.25, null_grad=0.2, null_step=20.,
             indices_list={}, snrv_list={}):
    """
    Input:  rho_coh_list: list of numpy arrays of coherent snr triggers
            rho_coinc_list: list of numpy arrays of coincident snr triggers
            null_min: Any trigger with null snr below this is cut
            null_grad: Any trigger with null snr<(null_grad*rho_coh+null_min)
                       is cut
            null_step: The value for required for coherent snr to start
                       increasing the null threshold
            indices_list: Optional- list of arrays of indices of triggers. If given, will remove
                   triggers that fail cuts
            snrv_list: Optional- list of arrays of individual ifo snr for triggers. If given will
                  remove triggers that fail cut
    Output: null_list: list of arrays of null snr for surviving triggers
            rho_coh_list_new: list of arrays of coherent snr for surviving triggers
            rho_coinc_list_new: list of arrays of coincident snr for suviving triggers
            indices_list_new: list of arrays of indices for surviving triggers
            snrv_list_new: list of arrays of Single detector snr for surviving triggers
    """
    null_list=[]
    rho_coh_list_new=[]
    rho_coinc_list_new=[]
    indices_list_new=[]
    snrv_list_new=[]

    for rho_coh1,rho_coinc1,indices1,snrv1 in zip(rho_coh_list,rho_coinc_list,indices_list,snrv_list):
        null, rho_coh, rho_coinc, index, snrv = null_snr(rho_coh1,rho_coinc1,index=indices1,snrv=snrv1)
        null_list.append(null)
        rho_coh_list_new.append(rho_coh)
        rho_coinc_list_new.append(rho_coinc)
        indices_list_new.append(index)
        snrv_list_new.append(snrv)

    return null_list, rho_coh_list_new, rho_coinc_list_new, indices_list_new, snrv_list_new

def null_snr(rho_coh, rho_coinc, null_min=5.25, null_grad=0.2, null_step=20.,
             index={}, snrv={}):
    """
    Input:  rho_coh: Numpy array of coherent snr triggers
            rho_coinc: Numpy array of coincident snr triggers
            null_min: Any trigger with null snr below this is cut
            null_grad: Any trigger with null snr<(null_grad*rho_coh+null_min)
                       is cut
            null_step: The value for required for coherent snr to start
                       increasing the null threshold
            index: Optional- Indexes of triggers. If given, will remove
                   triggers that fail cuts
            snrv: Optional- Individual ifo snr for triggers. If given will
                  remove triggers that fail cut
    Output: null: null snr for surviving triggers
            rho_coh: Coherent snr for surviving triggers
            rho_coinc: Coincident snr for suviving triggers
            index: Indexes for surviving triggers
            snrv: Single detector snr for surviving triggers
    """
    null2 = rho_coinc**2 - rho_coh**2
    # Numerical errors may make this negative and break the sqrt, so set
    # negative values to 0.
    null2[null2 < 0] = 0
    null = null2**0.5
    # Make cut on null.
    keep1 = np.logical_and(null < null_min, rho_coh <= null_step)
    keep2 = np.logical_and(null < (rho_coh * null_grad + null_min),
                          rho_coh > null_step)
    keep = np.logical_or(keep1, keep2)
    index = index[keep]
    rho_coh  = rho_coh[keep]
    snrv = {ifo : snrv[ifo][keep] for ifo in snrv.keys()}
    rho_coinc = rho_coinc[keep]
    null = null[keep]
    return null, rho_coh, rho_coinc, index, snrv

def get_coinc_indexes(idx_dict, time_delay_idx):
    """
    Input: idx_dict: Dictionary of indexes of triggers above threshold in
                     each detector
           time_delay_idx: Dictionary giving time delay index
                           (time_delay*sample_rate) for each ifo
    Output: coinc_idx: list of indexes for triggers in geocent time that
                       appear in multiple detectors
    """
    coinc_list = {}#np.array([], dtype=int)
    for ifo in idx_dict.keys():
        """
        Create list of indexes above threshold in single detector in
        geocent time. Can then search for triggers that appear in multiple
        detectors later.
        """
        if len(idx_dict[ifo]) != 0:
            if not isinstance(time_delay_idx[ifo],int):
                coinc_list[ifo] = np.vstack([idx_dict[ifo] - t for t in time_delay_idx[ifo]])
            else:
                coinc_list[ifo] =  np.vstack([idx_dict[ifo] - time_delay_idx[ifo]])
            #Search through coinc_idx for repeated indexes. These must have
            #been loud in at least 2 detectors.
    if not isinstance(time_delay_idx[ifo],int):
        coinc_idx = [np.unique(np.hstack([np.intersect1d(coinc_list[k1][i], coinc_list[k2][i]) \
            for k1,k2 in combinations(coinc_list.keys(),2)])) for i in range(len(time_delay_idx[ifo]))]
    else:
        coinc_idx = [np.unique(np.hstack([np.intersect1d(coinc_list[k1][0], coinc_list[k2][0]) \
            for k1,k2 in combinations(coinc_list.keys(),2)]))]
    return coinc_idx

