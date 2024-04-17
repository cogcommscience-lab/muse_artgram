import os.path as op

import mne
from mne.datasets import misc
from mne.io import RawArray
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from autoreject import get_rejection_threshold, AutoReject, compute_thresholds 


from collections import OrderedDict, defaultdict
import logging
from pathlib import Path
import numpy as np
logger = logging.getLogger()
import matplotlib.pyplot as plt


from mne.channels import read_custom_montage, make_standard_montage
from mne import create_info

def _interpolate(x: np.ndarray, y: np.ndarray, new_x: np.ndarray,
                 kind='linear') -> np.ndarray:
    '''Perform interpolation for _sync_timestamps
    If scipy is not installed, the method falls back to numpy, and then only
    supports linear interpolation.
    '''
    try:
        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind=kind, axis=0,
                     assume_sorted=True,  # speed up
                     bounds_error=False)
        return f(new_x)
    except ImportError as e:
        if kind != 'linear':
            raise e
        else:
            return np.interp(new_x, xp=x, fp=y, left=np.NaN, right=np.NaN)

        

def _sync_timestamps(streams, kind='linear'):
    '''Sync all streams to the fastest sampling rate by shifting or upsampling.
    Depending on a streams channel-format, extrapolation is performed using
    with NaNs (numerical formats) or with [''] (string format).
    Interpolation is only performed for numeric values, and depending on the
    kind argument which is inherited from scipy.interpolate.interp1d. Consider
    that If the channel format is an integer type (i.e. 'int8', 'int16',
    'int32', or 'int64'), integer output is enforced by rounding the values.
    Additionally, consider that not all interpolation methods are convex, i.e.
    for some kinds, you might receive values above or below the desired
    integer type. There is no correction implemented for this, as it is assumed
    this is a desired behavior if you give the appropriate argument.
    For string formats, events are shifted towards the nearest feasible
    timepoint. Any time-stamps without a marker get then assigned an empty
    marker, i.e. [''].
    '''
    # selecting the stream with the highest effective sampling rate
    srate_key = 'effective_srate'
    srates = [stream['info'][srate_key] for stream in streams]
    max_fs = max(srates, default=0)

    if max_fs == 0:  # either no valid stream or all streams are async
        return streams
    if srates.count(max_fs) > 1:
        # highly unlikely, with floating point precision and sampling noise
        # but anyways: better safe than sorry
        logger.warning('I found at least two streams with identical effective '
                       'srate. I select one at random for syncing timestamps.')

    # selecting maximal time range of the whole recording
    # streams with fs=0 might are not dejittered be default, and therefore
    # indexing first and last might miss the earliest/latest
    # we therefore take the min and max timestamp
    stamps = [stream['time_stamps'] for stream in streams]
    ts_first = min((min(s) for s in stamps))
    ts_last = max((max(s) for s in stamps))

    # generate new timestamps
    # based on extrapolation of the fastest timestamps towards the maximal
    # time range of the whole recording
    fs_step = 1.0/max_fs
    new_timestamps = stamps[srates.index(max_fs)]
    num_steps = int((new_timestamps[0]-ts_first)/fs_step) + 1
    front_stamps = np.linspace(ts_first, new_timestamps[0], num_steps)
    num_steps = int((ts_last-new_timestamps[-1])/fs_step) + 1
    end_stamps = np.linspace(new_timestamps[-1], ts_last, num_steps)

    new_timestamps = np.concatenate((front_stamps,
                                     new_timestamps[1:-1],
                                     end_stamps),
                                    axis=0)

    # interpolate or shift all streams to the new timestamps
    for stream in streams:
        channel_format = stream['info']['channel_format'][0]

        if ((channel_format == 'string') and (stream['info'][srate_key] == 0)):
            # you can't really interpolate strings; and streams with srate=0
            # don't have a real sampling rate. One approach to sync them is to
            # shift their events to the nearest timestamp of the new
            # timestamps
            shifted_x = []
            for x in stream['time_stamps']:
                argmin = (abs(new_timestamps-x)).argmin()
                shifted_x.append(new_timestamps[argmin])

            shifted_y = []
            for x in new_timestamps:
                try:
                    idx = shifted_x.index(x)
                    y = stream['time_series'][idx]
                    shifted_y.append([y])
                except ValueError:
                    shifted_y.append([''])

            stream['time_series'] = np.asanyarray((shifted_y))
            stream['time_stamps'] = new_timestamps

        elif channel_format in ['float32', 'double64', 'int8', 'int16',
                                'int32', 'int64']:
            # continuous interpolation is possible using interp1d
            # discrete interpolation requires some finetuning
            # bounds_error=False replaces everything outside of interpolation
            # zone with NaNs
            y = stream['time_series']
            x = stream['time_stamps']

            stream['time_series'] = _interpolate(x, y, new_timestamps, kind)
            stream['time_stamps'] = new_timestamps

            if channel_format in ['int8', 'int16', 'int32', 'int64']:
                # i am stuck with float64s, as integers have no nans
                # therefore i round to the nearest integer instead
                stream['time_series'] = np.around(stream['time_series'], 0)
        else:
            raise NotImplementedError("Don't know how to sync sampling for "
                                      "channel_format="
                                      "{}".format(channel_format))
        stream['info']['effective_srate'] = max_fs

    return streams


def _limit_streams_to_overlap(streams):
    '''takes streams, returns streams limited to time periods overlapping
    between all streams
    The overlapping periods start and end for each streams with the first and
    last sample completely within the overlapping period.
    If time_stamps have been synced, these are the same time-points for all
    streams. Consider that in the case of unsynced time-stamps, the time-stamps
    can not be exactly equal!
    '''
    ts_first, ts_last = [], []
    for stream in streams:
        # skip streams with fs=0 or if they send strings, because they might
        # just not yet have send anything on purpose (i.e. markers)
        # while other data was already  being recorded.
        if (stream['info']['effective_srate'] != 0 and
            stream['info']['channel_format'][0] != 'string'):
            # extrapolation in _sync_timestamps is done with NaNs
            not_extrapolated = np.where(~np.isnan(stream['time_series']))[0]
            ts_first.append(min(stream['time_stamps'][not_extrapolated]))
            ts_last.append(max(stream['time_stamps'][not_extrapolated]))

    ts_first = max(ts_first)
    ts_last = min(ts_last)
    for stream in streams:
        # use np.around to prevent floating point hickups
        around = np.around(stream['time_stamps'], 15)
        a = np.where(around >= ts_first)[0]
        b = np.where(around <= ts_last)[0]
        select = np.intersect1d(a, b)
        if type(stream['time_stamps']) is list:
            stream['time_stamps'] = [stream['time_stamps'][s] for s in select]
        else:
            stream['time_stamps'] = stream['time_stamps'][select]

        if type(stream['time_series']) is list:
            stream['time_series'] = [stream['time_series'][s] for s in select]
        else:
            stream['time_series'] = stream['time_series'][select]

    return streams


def preprocessing_autoreject(stream):
    data = stream["time_series"][10:-10].T
    assert data.shape[0] == 4  # four raw EEG plus one stim channel

    data *= 1e-6
    sfreq = float(stream["info"]["nominal_srate"][0])
    ch_names = ["TP9", "AF7", "AF8", "TP10"]
    ch_types  = ['eeg']*4
    montage = make_standard_montage('standard_1005')

    info = create_info(ch_names=ch_names, ch_types=ch_types,
                               sfreq=sfreq)

    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)

    raw.filter(0.1, 20, method='iir', verbose= False)
    epochs_raw = mne.make_fixed_length_epochs(raw, duration=4, preload=True)
    epochs = epochs_raw.copy()


    reject = get_rejection_threshold(epochs, random_state = 42, decim=2)

    epochs.drop_bad(reject=reject)


    ar = AutoReject(n_interpolate=np.arange(5), random_state=11,
                               n_jobs=1, verbose=True)
    ar.fit(epochs)  # fit on a few epochs to save time
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    timeseries_df = epochs_ar.to_data_frame()[['epoch','time', 'TP9', 'AF7', 'AF8', 'TP10']]

    full_epoch_length = epochs_raw.get_data().shape[0]
    num_channels = epochs_raw.get_data().shape[1]
    num_timesteps_per_epoch = epochs_raw.get_data().shape[2]

    epochs_dropped = np.array(list(set(np.arange(full_epoch_length)).difference(timeseries_df.epoch.drop_duplicates())))
    num_epoch_dropped = len(epochs_dropped)

    zero_epochs_df = pd.DataFrame({"epoch":epochs_dropped.repeat(num_timesteps_per_epoch),
                                   "time":np.tile(timeseries_df.time.drop_duplicates(),num_epoch_dropped),
                 "TP9":np.zeros(num_timesteps_per_epoch*num_epoch_dropped),
                 "AF7":np.zeros(num_timesteps_per_epoch*num_epoch_dropped),
                 "AF8":np.zeros(num_timesteps_per_epoch*num_epoch_dropped),
                 "TP10":np.zeros(num_timesteps_per_epoch*num_epoch_dropped)})

    timeseries_df_concated = pd.concat([timeseries_df, zero_epochs_df]).sort_values(by=['epoch','time']).reset_index()
    
    return(timeseries_df_concated)