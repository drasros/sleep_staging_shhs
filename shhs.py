import os
import pickle
import glob

import re

import pyedflib

import numpy as np
import scipy.signal as ssignal

from tqdm import tqdm
from sklearn import preprocessing

# NOTE: Does not work on SHHS2 yet...
# SHHS2 has a mix of 125Hz records and 128Hz records. 
# TODO: deal with this... (resample or use adaptive model ?)

def read_edf_and_annot(edf_file, hyp_file, channel='EEG', shhs='1'):
    rawsignal = read_edfrecord(edf_file, channel, shhs)
    stages = read_annot_regex(hyp_file)
    print(len(stages))
    print('rawsignal ', len(rawsignal) / (30*125))
    # check that they have the same length
    if shhs=='1':
        assert len(rawsignal) % (30*125) == 0.
        assert int(len(rawsignal) / (30*125)) == len(stages)
    else:
        assert len(rawsignal) % (30*128) == 0.
        assert int(len(rawsignal) / (30*128)) == len(stages)
    return rawsignal, np.array(stages)

def read_annot_regex(filename):
    with open(filename, 'r') as f:
        content = f.read()
    # Check that there is only one 'Start time' and that it is 0
    patterns_start = re.findall(
        r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', 
        content)
    assert len(patterns_start) == 1
    # Now decode file: find all occurences of EventTypes marking sleep stage annotations
    patterns_stages = re.findall(
        r'<EventType>Stages.Stages</EventType>\n' +
        r'<EventConcept>.+</EventConcept>\n' +
        r'<Start>[0-9\.]+</Start>\n' +
        r'<Duration>[0-9\.]+</Duration>', 
        content)
    print(patterns_stages[-1])
    stages = []
    starts = []
    durations = []
    for pattern in patterns_stages:
        lines = pattern.splitlines()
        stageline = lines[1]
        stage = int(stageline[-16])
        startline = lines[2]
        start = float(startline[7:-8])
        durationline = lines[3]
        duration = float(durationline[10:-11])
        assert duration % 30 == 0.
        epochs_duration = int(duration) // 30

        stages += [stage]*epochs_duration
        starts += [start]
        durations += [duration]
    # last 'start' and 'duration' are still in mem
    # verify that we have not missed stuff..
    assert int((start + duration)/30) == len(stages)
    return stages

def read_edfrecord(filename, channel, shhs='1'):

    # SHHS1 is at 125 Hz, SHHS2 at 128Hz

    assert channel=='EEG' or channel=='EEG(sec)', "Channel must be EEG or EEG(sec)"
    f = pyedflib.EdfReader(filename)
    print("Startdatetime: ", f.getStartdatetime())
    signal_labels = f.getSignalLabels()
    assert channel in signal_labels
    idx_chan = [i for i, x in enumerate(signal_labels) if x==channel][0]
    sigbuf = np.zeros((f.getNSamples()[idx_chan]))
    sigbuf[:] = f.readSignal(idx_chan)
    samplingrate = len(sigbuf) / f.file_duration
    print("sampling rate: ", samplingrate)
    if shhs=='1':
        print("30s * 125Hz divides signal length ?: ", len(sigbuf)%(30*125)==0)
        assert samplingrate == 125., "Sampling rate is not 125Hz on this record"
    else:
        print("30s * 128Hz divides signal length ?: ", len(sigbuf)%(30*128)==0)
        assert samplingrate == 128., "Sampling rate is not 128Hz on this record"
    return sigbuf


def prepare_dataset(shhs_base_dir, preprocessed_dir=None,
                    shhs='1', nb_patients=None, 
                    filter=True, channel='EEG'):
    assert shhs in ['1', '2']
    # shhs1 is as 125Hz
    # shhs2 is at 128Hz
    # if filter, in both cases it will be at 0.5-25Hz
    # + subsampling / 2: 64 or 62.5Hz, i.e. 1920 or 1875 points per 30s segment. 

    # read and preprocessed data will be saved in
    # preprocessed_dir
    fil = 'filtered' if filter else 'nofilter'

    if preprocessed_dir is None:
        preprocessed_dir = os.path.join(shhs_base_dir, 'preprocessed', 'shhs'+shhs, fil, channel)

    edf_dir = os.path.join(shhs_base_dir, 'polysomnography', 'edfs', 'shhs'+shhs)
    print(edf_dir)
    edf_names = glob.glob(os.path.join(edf_dir, '*.edf'))
    
    hyp_dir = os.path.join(shhs_base_dir, 'polysomnography', 'annotations-events-nsrr', 'shhs'+shhs)
    print(hyp_dir)
    hyp_names = glob.glob(os.path.join(hyp_dir, 'shhs*.xml'))

    print("number of records: ", len(edf_names))
    print("number of hypnograms", len(hyp_names))

    if nb_patients is None:
        nb_patients = len(edf_names)
    print('Number of patients: ', nb_patients)

    # loop on records
    for i in range(nb_patients):
        name = os.path.basename(edf_names[i])
        name = name[:-4]
        hyp_try_name = os.path.join(hyp_dir, name + '-nsrr.xml')
        save_file = os.path.join(preprocessed_dir, name + '.p')
        if not os.path.exists(save_file):
            try:
                print('#####################################')
                print('loading data and annotations for patient %d: ' %i),
                print(edf_names[i])
                rawsignal, stages = read_edf_and_annot(
                    edf_names[i], hyp_try_name, channel=channel, shhs=shhs)
                # original labels:
                # Wake: 0
                # Stage 1: 1
                # Stage 2: 2
                # Stage 3: 3
                # Stage 4: 4
                # REM: 5
                # verify that there is no other stage
                labels, counts = np.unique(stages, return_counts=True)
                print('Labels and counts: '),
                print(labels, counts)
                assert np.max(labels) <= 5

                print('channel used: ', channel)
                samplesper30s = 30 * 125 if shhs=='1' else 30*125
                if filter:
                    lowcut = 0.5
                    highcut = 25
                    nyquist_freq = 128 / 2.
                    low = lowcut / nyquist_freq
                    high = highcut / nyquist_freq
                    b, a = ssignal.butter(3, [low, high], btype='band')
                    rawsignal = ssignal.filtfilt(b, a, rawsignal)
                    # subsample (with average)
                    rawsignal = ssignal.resample(rawsignal, len(rawsignal)//2)
                    samplesper30s /= 2


                # and now rearrange into nparray of training samples
                numberofintervals = rawsignal.shape[0] / samplesper30s
                examples_patient_i = np.array(
                    np.split(rawsignal, numberofintervals, axis=0))
                print('examples.shape for patient %d : ' %i),
                print(examples_patient_i.shape)
                print('stages.shape for patient %d : ' %i),
                print(stages.shape)

                # Remove excess 'pre-sleep' and 'post-sleep' wake
                # so that total 'out of night' wake is at most equal 
                #to the biggest other class
                if stages[np.argmax(counts)]==0:
                    print('Wake is the biggest class. Trimming it..')
                    second_biggest_count = np.max(
                        np.delete(counts, np.argmax(counts)))
                    occurencesW = np.where(stages==0)[0]
                    last_W_evening_index = min(
                        np.where(occurencesW[1:] - occurencesW[0:-1]!=1)[0])
                    nb_evening_Wepochs = last_W_evening_index + 1
                    first_W_morning_index = len(stages) 
                    - min(np.where(
                        (occurencesW[1:]-occurencesW[0:-1])[::-1]!=1)[0]) - 1
                    nb_morning_Wepochs = len(stages) - first_W_morning_index
                    nb_pre_post_sleep_wake_eps = \
                    nb_evening_Wepochs + nb_morning_Wepochs
                    print('number of pre and post sleep wake epochs: '),
                    print(nb_pre_post_sleep_wake_eps)
                    if nb_pre_post_sleep_wake_eps > second_biggest_count:
                        total_Weps_to_remove = nb_pre_post_sleep_wake_eps - second_biggest_count
                        print('removing %i wake epochs' %total_Weps_to_remove)
                        if nb_evening_Wepochs > total_Weps_to_remove:
                            stages = stages[total_Weps_to_remove:]
                            examples_patient_i = examples_patient_i[total_Weps_to_remove:, :]
                        else:
                            evening_Weps_to_remove = nb_evening_Wepochs
                            morning_Weps_to_remove = total_Weps_to_remove - evening_Weps_to_remove
                            stages = stages[evening_Weps_to_remove:-morning_Weps_to_remove]
                            examples_patient_i = examples_patient_i[evening_Weps_to_remove:-morning_Weps_to_remove, :]
                else:
                    print('Wake is not the biggest class, nothing to remove.')

                # merge labels 3 and 4
                indices = np.where(stages==4)
                stages[indices] = 3
                # now use label 4 for REM
                le = preprocessing.LabelEncoder()
                le.fit([0, 1, 2, 3, 5])
                stages = le.transform(stages)

                # show counts
                cl, cnts = np.unique(stages, return_counts=True)
                print('Labels and counts after W rebalance and merge 3-4: ', cl, cnts)

                # pickle this patient data to disk
                data = examples_patient_i, stages
                print('saving..')
                if not os.path.exists(os.path.dirname(save_file)):
                    os.makedirs(os.path.dirname(save_file))
                with open(save_file, 'wb') as fp:
                    dataset = pickle.dump(data, fp)
                
            except FileNotFoundError:
                print('File not found.')
            except AssertionError:
                print('AssertionError. This patient probably has more than 0 to 5 labels...')
                print('Skipping this patient.')
            finally:
                pass

def batches_iterator(iterator, batch_size):
    while True:
        examples = []
        labels_target = []

        b_elems = 0
        while b_elems < batch_size:
            expl, lbl_t = next(iterator)
            examples += [expl]
            labels_target += [lbl_t]
            b_elems += 1
        examples = np.array(examples)
        labels_target = np.squeeze(np.array(labels_target))

        yield examples, labels_target


def data_iterator_fixedN(preprocessed_names, n, n_patient_queues,
                         epochs_before, epochs_after,
                         balance_classes=True,
                         use_if_missing_stage=False):
    examples_trained = 0
    while examples_trained < n:
        names_iterator = patient_names_iterator(preprocessed_names)
        it_1ep = data_iterator_1epoch(names_iterator, n_patient_queues,
            epochs_before, epochs_after, balance_classes, use_if_missing_stage)
        try:
            while examples_trained < n:
                res = next(it_1ep)
                examples_trained += 1
                yield res
        except StopIteration:
            pass
        finally:
            del it_1ep


def data_iterator_1epoch(patient_names_iterator, n_patient_queues, 
                         epochs_before, epochs_after, 
                         balance_classes=True, use_if_missing_stage=False):

    # problem somewhere with this function
    # it yields a number of example proportional to nb_patient_queues
    # this is not normal
    # probably what happens is that the iterators_1p share a same stop
    # signal and stop too early..
    # try to implement with multiprocessing

    iterators_1p = [data_iterator_1p(next(patient_names_iterator),
                                     epochs_before, epochs_after, 
                                     balance_classes, use_if_missing_stage) 
                    for _ in range(n_patient_queues)]
    iterators_1p_finished = []

    try:
        while True:
            while len(iterators_1p_finished) < n_patient_queues:
                for r in range(n_patient_queues):
                    if r not in iterators_1p_finished:
                        try:
                            res = next(iterators_1p[r])
                            yield res
                        except StopIteration:
                            try:
                                a = iterators_1p[r]
                                name = next(patient_names_iterator)
                                iterators_1p[r] = data_iterator_1p(
                                    name, 
                                    epochs_before, epochs_after, 
                                    balance_classes, use_if_missing_stage)
                                del a
                                try:
                                    # In theory this should not happen.
                                    # However, I got fooled by the following:
                                    # if use_if_missing_stage is false, an iterator_1p
                                    # gives a StopIteration the first time it is called!!!
                                    # and it stops the whole loop...
                                    res = next(iterators_1p[r])
                                    # should always be possible because it was just created
                                    yield res
                                except StopIteration:
                                    pass

                            except StopIteration:
                                # no more patient_names for this epoch
                                #print('No more patient names. Removing one queue. ')
                                iterators_1p_finished += [r]

            # after all patient queues are empty, 
            # catch a StopIteration to finish. 
            _ = next(iterators_1p[0])
    except StopIteration:
        print('DONE ITERATING ONE EPOCH.')
    finally:
        del patient_names_iterator


def patient_names_iterator(preprocessed_names):
    mix = np.arange(len(preprocessed_names))
    np.random.shuffle(mix)
    for i in range(len(preprocessed_names)):
        yield preprocessed_names[mix[i]]

def data_iterator_1p(preprocessed_name, epochs_before, epochs_after, 
                     balance_classes=True, use_if_missing_stage=False,
                     shuffle_within_patient=True):
    # Note: even with the 'balanced=True' and 'use_if_missing_stage'
    # options, there can be be a slight imbalance
    # due to samples not yielded when incomplete=True. This is normal. 

    totalyielded = 0

    with open(preprocessed_name, 'rb') as fp:
        data = pickle.load(fp)
    epochs, stages = data
    stages = np.array(stages)

    cl, cnts = np.unique(stages, return_counts=True)
    nb_cl = len(cl)
    cl_is_5 = nb_cl==5
    yield_or_not = True if use_if_missing_stage else cl_is_5
    indices_per_class = [np.where(stages==i)[0] for i in range(np.max(cl)+1)]
    for r in range(np.max(cl)+1):
        np.random.shuffle(indices_per_class[r])
    if balance_classes == True:
        min_cnt = np.min(cnts)
        indices_to_yield = [a[:min_cnt] for a in indices_per_class]
    else:
        indices_to_yield = indices_per_class
    indices_to_yield = np.concatenate(indices_to_yield)

    if shuffle_within_patient:
        np.random.shuffle(indices_to_yield)

    nb_ex_to_yield = len(indices_to_yield)

    for i in range(nb_ex_to_yield):
        idx = indices_to_yield[i]
        res = read_one_idx(epochs, stages, idx, epochs_before, epochs_after)
        if res is not None and yield_or_not:
            #print(totalyielded)
            totalyielded += 1
            yield res

def read_one_idx(epochs, stages, idx, epochs_before, epochs_after):
    examples_before = []
    examples_after = []
    incomplete = False
    for ep_b in range(epochs_before):
        try:
            examples_before += [epochs[idx - epochs_before + ep_b]]
        except IndexError:
            incomplete=True
    # epochs_after can be an int or a real between 0 and 1
    for ep_a in range(int(epochs_after)):
        try:
            examples_after += [epochs[idx + 1 + ep_a]]
        except IndexError:
            incomplete=True
    # if epochs after is not int, add the remaining samples
    if isinstance(epochs_after, float):
        try:
            int_part = int(epochs_after)
            decimal_part = epochs_after - int(epochs_after)
            samples_to_keep = int(decimal_part * 1875)
            examples_after += [epochs[idx + 1 + int(epochs_after)][:samples_to_keep]]
        except IndexError:
            incomplete=True

    example = epochs[idx]

    examples = examples_before + [example] + examples_after
    examples = [np.array(ex) for ex in examples]
    example_joint = np.concatenate(examples)
    label_target = stages[idx]

    if not incomplete:
        return example_joint, np.eye(5)[label_target]
    else: return None







