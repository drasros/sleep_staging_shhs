import numpy as np

from model import CNN

import shhs

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

############################################
# CONFIG

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
### Save dir 
save_dir = '/datadrive1/hyp/'
### Load_dir. This is is where checkpoint, model.ckpt.data-00000-of-00001 etc are present
load_dir = '/datadrive1/tmp10/exp_300000_umTrue_hl100/best'
patientlist_dir = '/datadrive1/exp10/exp_300000_umTrue_hl100'
# data
shhs_base_dir = '/datadrive1/data/shhs'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
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

# read first patient name
with open(os.path.join(patientlist_dir, 'names_test.txt'), 'r') as f:
    l = f.readline()[:-1]

patient_to_plot = os.path.join(shhs_base_dir, 'preprocessed', 'shhs1', 
                                'nofilter', 'EEG', l)# + '.p')
print(patient_to_plot)

it_p = shhs.data_iterator_1p_ordered(
    patient_to_plot, 2, 1, False, 
    use_if_missing_stage=False) # modify if needed

target_cl = []
pred_cl = []


try:
    while True:
        ex, cl = next(it_p)
        numeric_in = {
            'inX': np.tile(ex, [128, 1]),
            'targetY': np.tile(cl, [128, 1]),
            'phase': 0,
        }
        pred_val, acc_val, _ = model.estimate_model(numeric_in)
        pred_val = pred_val[0]
        # simpler than batching and de-batching
        target_cl += [cl]
        pred_cl += [pred_val]
except StopIteration:
    print('done')
finally:
    pass

target_cl = np.argmax(target_cl, axis=1)
pred_cl = np.array(pred_cl)
print(target_cl)
print(pred_cl)

np.save(os.path.join(save_dir, 'target.npy'), target_cl)
np.save(os.path.join(save_dir, 'pred.npy'), pred_cl)

plt.plot(range(len(target_cl)), target_cl)
plt.savefig(os.path.join(save_dir, 'hyp_target.eps'), format='eps', dpi=300)
plt.close()

plt.plot(range(len(pred_cl)), pred_cl)
plt.savefig(os.path.join(save_dir, 'hyp_pred.eps'), format='eps', dpi=300)
plt.close()

plt.plot(range(len(target_cl)), target_cl, range(len(target_cl)), pred_cl)
plt.savefig(os.path.join(save_dir, 'hyp.eps'), format='eps', dpi=300)






