import numpy as np

from model import CNN

import sys
sys.path.append('/home/arnaud/data_these/tensorflow1/tflib/')

import ops
import os
import tensorflow as tf
import argparse

# before importing pyplot, choose backend. 
# This allows use on server where $DISPLAY env var is not defined
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib2tikz import save as tikz_save

############################################
# CONFIG

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
### Save dir 
save_dir = '/datadrive1/viz/'
### Load_dir. This is is where checkpoint, model.ckpt.data-00000-of-00001 etc are present
load_dir = '/datadrive1/tmp/best'
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

##### Model
batch_size = 128
featuremap_sizes = [128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256]
strides = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
filter_sizes = [7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 3, 3]
hiddenlayer_size = 100
balance_sm = [1, 1, 1, 1, 1]
balance_cost = False
eps_before = 2
eps_after = 1
filter_ = False

############################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-nb_iterations', type=int, default=500)
parser.add_argument('-learning_rate', type=float, default=0.01)
parser.add_argument('-normalize_grads', type=str2bool, default=str2bool('True'))
args = parser.parse_args()

nb_iterations = args.nb_iterations
step = args.learning_rate
normalize_grads = args.normalize_grads

this_dir = 'step%.3f_its%d_norm%s' %(step, nb_iterations, str(normalize_grads))
save_dir = os.path.join(save_dir, this_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
############################################

print('CREATING MODEL')
model = CNN(batch_size=batch_size,
            featuremap_sizes=featuremap_sizes,
            strides=strides,
            filter_sizes=filter_sizes,
            hiddenlayer_size=hiddenlayer_size,
            balance_sm=balance_sm,
            balance_cost=balance_cost,
            eps_before=eps_before,
            eps_after=eps_after,
            filter=filter_)
model.init()

# load model
# lastly trained model:
model.load_model(load_dir)#/checkpoints')
#####################

for cl in range(5):
    scale = 1
    costs = []
    # Start with a null signal + some noise
    # amplitude of the real data on eef is mostly within [-160, 160] microV
    # let's use 10 microV for noise variance
    signal = np.random.normal(loc=0, scale=scale, size=(batch_size, 3750*(eps_before+1+eps_after)))
    # note: we are feeding 4 consecutive epochs (2 before and 1 after)

    # Now do the optimization to maximize class activation
    for epoch in range(nb_iterations):
        numeric_in = {
            'inX': signal.astype(np.float32),
            'cl_viz': cl,
        }
        cost_numerical, grads_numerical = model.get_input_grads(numeric_in)
        # trick: normalize gradients (whithin examples) to make grad ascent smoother
        if normalize_grads:
            grads_numerical /= np.sqrt(np.mean(np.square(grads_numerical), 
                                       axis=1, 
                                       keepdims=True)) + 1e-5
        print('iteration', epoch, " , cost ", cost_numerical)
        costs += [cost_numerical]
        signal += grads_numerical * step
        mean = np.mean(signal)
        # keep variance equal to 1
        signal /= np.mean(np.square(signal-mean)) + 1e-5

    # plot costs to see whether optimization is complete
    plt.plot(costs)
    plt.savefig(os.path.join(save_dir, 'costs_stage%d' %cl))
    plt.close()

    x = np.arange(3750*(eps_before+1+eps_after))
    # apply low-pass filter 1-30Hz
    # from scipy.signal import kaiserord, lfilter, firwin
    # nyq_rate = 125. / 2
    # width = 5. / nyq_rate
    # ripple_db = 60.
    # N, beta = kaiserord(ripple_db, width)
    # cutoff_hz = 30.
    # taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    # signal = lfilter(taps, 1.0, signal)

    # visualization is not very nice when we keep all frequencies
    # because a lot of high-freq is allowed
    # apply butterworth band-pass filter 1-30Hz (EEG freqs)
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * 125.
    low = 1. / nyq
    high = 30. / nyq
    order = 6
    b, a = butter(order, [low, high], btype='band')
    signal = filtfilt(b, a, signal)

    #plt.plot(x/125., signal[0, :])
    #plt.xlabel('time (s)')
    #plt.ylabel('amplitude\n(arbitrary\nunit)')
    #plt.savefig(os.path.join(save_dir, 'viz_out_stage%s.png' %str(cl)))
    #tikz_save(os.path.join(save_dir, 'viz_out_stage%s.tex' %str(cl)))
    #plt.close()

    # actual epoch considered
    x_ = x[3750*eps_before:3750*(eps_before+1)]
    signal_ = signal[:, 3750*eps_before:3750*(eps_before+1)]
    x_ = np.arange(len(x_))
    # plot seconds 5 to 15 for better readability
    x__ = x_[125*5:125*15]
    signal__ = signal_[:, 125*5:125*15]
    for i_b in range(10):
        plt.figure(figsize=(20, 3))
        plt.plot(x__/125., signal__[i_b, :])
        plt.xlim([5, 15])
        plt.xlabel('time(s)', fontsize=22)
        plt.ylabel('amplitude', fontsize=22)
        plt.tick_params(axis='both', labelsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'viz_out_stage%s_10s_%d.png' %(str(cl), i_b)))
        plt.savefig(os.path.join(save_dir, 'viz_out_stage%s_10s_%d.eps' %(str(cl), i_b)), format='eps', dpi=300)
        tikz_save(os.path.join(save_dir, 'viz_out_stage%s_10s_%d.tex' %(str(cl), i_b)))
        plt.close()







