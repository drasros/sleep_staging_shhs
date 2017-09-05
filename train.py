import numpy as np
import tensorflow as tf
import sklearn.metrics

import argparse
import time
import os
import glob
import sys

import utils

from tqdm import tqdm

import shhs

import ops

from model import CNN

##########################################################
############  CONFIG #####################################
shhs_base_dir = '/datadrive1/data/shhs'

exp_num = '12'
results_dir = '/datadrive1/exp' + exp_num + '/'

checkpoint_dir = '/datadrive1/tmp'+exp_num+'/'
# checkpoint_dir_best = checkpoint_dir + 'best/'
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
# if not os.path.exists(checkpoint_dir_best):
#     os.makedirs(checkpoint_dir_best)
##########################################################

def write_to_comment_file(comment_file, text):
    with open(comment_file, "a") as f:
        f.write(text)

def write_name_list_to_file(file, name_list):
    with open(file, "a") as f:
        for item in name_list:
            f.write("%s\n" % item)

def get_lines_list(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        return lines

# CAUTION: make sure you filesystem supports long filenames (>256chars)
# def get_exp_name(nb_patients, batch_size, training_batches, learning_rates, conv_type, batch_norm, 
#                  activations, featuremap_sizes, strides, filter_sizes, 
#                  hiddenlayer_size, eps_before, eps_after, channel,
#                  balance_sm, balance_cost, balance_classes, use_if_missing_stage, filter):
#     # careful: no brackets in name! unroll lists
#     # also: careful to not use (total) path longer than 256 characters on NTFS volumes!!
#     # TODO: refine this to avoid forbidden characters ([, ] etc...)
#     name = "exp" + "_np" + str(nb_patients) + "_b" + str(batch_size) \
#            + "_lr" + str(*learning_rates) + "_" + str(*training_batches) \
#            + "_c" + str(conv_type) + "_bn" +str(batch_norm) \
#            + "_ac" + str(activations) + "_ar" + str(featuremap_sizes).replace(" ", "") \
#            + "_hl" + str(hiddenlayer_size) \
#            + "_eb" + str(eps_before) \
#            + "_ea" + str(eps_after) + "_" + str(channel)[4:] \
#            + "_bsm" + str(balance_sm).replace(" ", "") + "_bc" + str(balance_cost) \
#            + "_bcl" + str(balance_classes) + "_um" + str(use_if_missing_stage) \
#            + "_fi" + str(filter)
#     return name

def get_exp_name(nb_patients, batch_size, training_batches, learning_rates, conv_type, batch_norm, 
                 activations, featuremap_sizes, strides, filter_sizes, 
                 hiddenlayer_size, eps_before, eps_after, channel,
                 balance_sm, balance_cost, balance_classes, use_if_missing_stage, filter):
    name = "exp_" + str(*training_batches) \
           + "_um" + str(use_if_missing_stage) \
           + "_hl" + str(hiddenlayer_size)
    return name


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_patients', type=int, default=-1)
    parser.add_argument('-training_batches', nargs='+', type=int, 
                        default=[50000]) #int(10000))
    parser.add_argument('-learning_rates', nargs='+', type=float,
                        default=[1e-3])
    parser.add_argument('--checkpoint_ep_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--activations', type=str, default="lrelu")
    parser.add_argument('--conv_type', type=str, default="std")
    parser.add_argument('--batch_norm', type=str2bool, default=str2bool('False'))
    parser.add_argument('-featuremap_sizes', nargs='+', type=int, 
                        default=[16, 32, 64, 128, 128])
    parser.add_argument('-strides', nargs='+', type=int, 
                        default=[2, 2, 2, 2, 2])
    parser.add_argument('-filter_sizes', nargs='+', type=int,
                        default=[7, 7, 7, 7, 7])
    parser.add_argument('--hiddenlayer_size', type=int, default=100)
    parser.add_argument('--balance_sm', nargs='+', type=int, 
                        default=[1, 1, 1, 1, 1])
    parser.add_argument('--balance_cost', type=str2bool, default=str2bool('False'))
    parser.add_argument('--eps_before', type=int, default=2)
    parser.add_argument('--eps_after', type=float, default=1)
    parser.add_argument('--channel', type=str, default='EEG')
    parser.add_argument('--balance_classes', type=str2bool, default=str2bool('False'))
    parser.add_argument('--use_if_missing_stage', type=str2bool, default=str2bool('False'))
    parser.add_argument('--filter', type=str2bool, default=str2bool('True'))
    parser.add_argument('--n_patient_queues', type=int, default=50)
    # ALSO ADD LEARNING RATE SCHEDULE
    # (add placeholder in model...)
    args = parser.parse_args()
    train(args)


def train(args, tvt_counts=None):
    print(args)

    activations = ops.lrelu if args.activations=="lrelu" else tf.nn.relu

    nb_patients = None if args.nb_patients==-1 else args.nb_patients
    if nb_patients is not None:
        assert(args.n_patient_queues <= int(0.2 * nb_patients)), "Too many patient queues for the number of patient used"

    # if balancing classes samples, do not balance cost
    assert not (args.balance_classes and args.balance_cost), "It is probably a bad idea to both balance training samples over classes and weight the cost. "

    ########## prepare data ##############
    print("PREPARING DATA...")
    shhs.prepare_dataset(shhs_base_dir, nb_patients=nb_patients,
                        filter=args.filter, channel=args.channel)
    fil = 'filtered' if args.filter else 'nofilter'
    

    ######## prepare experiment ############
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    exp_name = get_exp_name(
        args.nb_patients, 
        args.batch_size, args.training_batches, args.learning_rates, 
        args.conv_type, args.batch_norm, 
        args.activations, args.featuremap_sizes, args.strides, 
        args.filter_sizes, args.hiddenlayer_size, 
        args.eps_before, args.eps_after, 
        args.channel, args.balance_sm, 
        args.balance_cost, args.balance_classes, 
        args.use_if_missing_stage, args.filter)

    exp_dir = os.path.join(results_dir, exp_name)
    exp_checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
    exp_checkpoint_dir_best = os.path.join(checkpoint_dir, exp_name, 'best')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(exp_checkpoint_dir):
        os.makedirs(exp_checkpoint_dir)
    if not os.path.exists(exp_checkpoint_dir_best):
        os.makedirs(exp_checkpoint_dir_best)

    comment_text = ("Training batches: " + str(args.training_batches) + '\n' +
                    "Batch size: " + str(args.batch_size) + '\n' +
                    "Learning rates: " + str(args.learning_rates) + '\n' +
                    "Featuremap_sizes: " + str(args.featuremap_sizes) + '\n' +
                    "Filter sizes: " + str(args.filter_sizes) + '\n' +
                    "Strides: " + str(args.strides) + '\n' +
                    "Hidden layer size: " + str(args.hiddenlayer_size) + '\n' +
                    "Conv type: " + str(args.conv_type) + '\n' +
                    "Batch norm: " + str(args.batch_norm) + '\n' +
                    "Activations: " + str(args.activations) + '\n' +
                    "Number of epochs before: " + str(args.eps_before) + '\n' +
                    "Number of epochs after: " + str(args.eps_after) + '\n' +
                    "Channel used: " + args.channel + '\n' + 
                    "Balance softmax: " + str(args.balance_sm) + '\n' +
                    "Balance cost: " + str(args.balance_cost) + '\n' +
                    "Balance classes: " + str(args.balance_classes) + '\n' +
                    "Use_if_missing_stage: " + str(args.use_if_missing_stage) + '\n' +
                    "Filter: " + str(args.filter) + '\n' +
                    "Number of patient queues: " + str(args.n_patient_queues) + '\n')

    write_to_comment_file(os.path.join(exp_dir, "comment.txt"), comment_text)

    ########## prepare train, valid, test sets ##############
    #load previously saved names if a saved model exists
    if tf.train.get_checkpoint_state(exp_checkpoint_dir_best) is not None:
        print("loading train/valid/test split from saved model...")
        train_file = os.path.join(exp_dir, 'names_train.txt')
        valid_file = os.path.join(exp_dir, 'names_valid.txt')
        test_file = os.path.join(exp_dir, 'names_test.txt')

        names_train = get_lines_list(train_file)
        names_train = [os.path.join(shhs_base_dir, 'preprocessed', 'shhs1', fil, args.channel, nm) for nm in names_train]
        names_valid = get_lines_list(valid_file)
        names_valid = [os.path.join(shhs_base_dir, 'preprocessed', 'shhs1', fil, args.channel, nm) for nm in names_valid]
        names_test = get_lines_list(test_file)
        names_test = [os.path.join(shhs_base_dir, 'preprocessed', 'shhs1', fil, args.channel, nm) for nm in names_test]

    else:
        preprocessed_names = glob.glob(os.path.join(
            shhs_base_dir, 'preprocessed', 'shhs1', fil, args.channel, '*.p'))
        preprocessed_names = preprocessed_names[:nb_patients]
        # shuffle
        r = np.arange(len(preprocessed_names))
        np.random.shuffle(r)
        preprocessed_names = [preprocessed_names[i] for i in r]

        tvt_proportions = (0.5, 0.2, 0.3)
        n_train = int(tvt_proportions[0]*len(preprocessed_names))
        print('n_train: ', n_train)
        n_valid = int(tvt_proportions[1]*len(preprocessed_names))
        print('n_valid: ', n_valid)
        names_train = preprocessed_names[0:n_train]
        names_valid = preprocessed_names[n_train:n_train+n_valid]
        names_test = preprocessed_names[n_train+n_valid:]

        write_name_list_to_file(os.path.join(exp_dir, "names_train.txt"), names_train)
        write_name_list_to_file(os.path.join(exp_dir, "names_valid.txt"), names_valid)
        write_name_list_to_file(os.path.join(exp_dir, "names_test.txt"), names_test)


    
    ######### define model ##############
    print("DEFINING MODEL...")

    model = CNN(batch_size=args.batch_size,
                featuremap_sizes=args.featuremap_sizes,
                strides=args.strides,
                filter_sizes=args.filter_sizes,
                hiddenlayer_size=args.hiddenlayer_size,
                balance_sm=args.balance_sm,
                balance_cost=args.balance_cost,
                conv_type=args.conv_type,
                batch_norm=args.batch_norm,
                activations=activations,
                eps_before=args.eps_before,
                eps_after=args.eps_after,
                filter=args.filter)

    model.init()

    #load previously saved model if there is one
    if tf.train.get_checkpoint_state(exp_checkpoint_dir_best) is not None:
        model.load_model(exp_checkpoint_dir_best)
        #model.load_model(exp_dir)
        # also load sequence of previous costs
        costs = np.load(os.path.join(exp_checkpoint_dir_best, 'costs.npy')).tolist()
        costs_valid = np.load(os.path.join(exp_checkpoint_dir_best, 'costs_valid.npy')).tolist()
        accs = np.load(os.path.join(exp_checkpoint_dir_best, 'accs.npy')).tolist()
        accs_valid = np.load(os.path.join(exp_checkpoint_dir_best, 'accs_valid.npy')).tolist()
        # TODO: also save (and load) current_best_cost and current_best_acc
    else:
        costs = []
        costs_valid = []
        accs = []
        accs_valid = []

    ########## TRAIN ##############################
    print('TRAINING...')

    which_val_metric = 'acc' #'cost' # whether to monitor acc or cost for best model

    try:
        #estimate how many batches an epoch is
        # print('iterating once over data to count the number of train batches for this split...')
        # names_iterator_train = shhs.patient_names_iterator(names_train)
        # it_1ep_train = shhs.data_iterator_1epoch(
        #     names_iterator_train, 
        #     args.n_patient_queues,
        #     epochs_before=args.eps_before,
        #     epochs_after=args.eps_after,
        #     balance_classes=args.balance_classes,
        #     use_if_missing_stage=args.use_if_missing_stage)
        # ex_per_ep = 0
        # try:
        #     while True:
        #         _, _ = next(it_1ep_train)
        #         ex_per_ep += 1
        # except StopIteration:
        #     pass
        # finally:
        #     del names_iterator_train, it_1ep_train
        # batches_per_ep = ex_per_ep // args.batch_size
        # print("Number of batches per epoch: ", batches_per_ep)

        # simply initialize iterators with a very large number
        # of batches to make sure they do not run out
        # the count will be made in the training loop. 
        it_train = shhs.data_iterator_fixedN(
            preprocessed_names=names_train, 
            n=int(1e8)*args.batch_size,
            n_patient_queues=args.n_patient_queues,
            epochs_before=args.eps_before, 
            epochs_after=args.eps_after, 
            balance_classes=args.balance_classes, 
            use_if_missing_stage=args.use_if_missing_stage)
        b_it_train = shhs.batches_iterator(it_train, args.batch_size)

        it_valid = shhs.data_iterator_fixedN(
            preprocessed_names=names_valid, 
            n=int(1e8)*args.batch_size,
            n_patient_queues=args.n_patient_queues,
            epochs_before=args.eps_before, 
            epochs_after=args.eps_after, 
            balance_classes=args.balance_classes, 
            use_if_missing_stage=args.use_if_missing_stage)
        b_it_valid = shhs.batches_iterator(it_valid, args.batch_size)

        # Training loop.
        # Validation cost is evaluated continuously, 
        # one batch every 5 train batches

        current_best_cost = 1e20
        current_best_acc = 0.

        for b in range(np.sum(args.training_batches)): #tqdm(range(np.sum(args.training_batches))):

            if b % 200 == 0:
                print(str(100*b/np.sum(args.training_batches)) + ' percent done...')
                sys.stdout.flush()

            # if b % (np.sum(args.training_batches)//10) == 0: # practical when working with server
            #     write_to_comment_file(os.path.join(exp_dir, 'STATUS.txt'), 
            #                           '%s percent done...' %(10*(b//(np.sum(args.training_batches)//10))))

            # For variable learning rate:
            c01 = b > np.cumsum(args.training_batches)
            idx_lr = np.where(c01==1)[0]
            if len(idx_lr) == 0:
                idx_lr = 0
            else:
                idx_lr = idx_lr[-1] + 1
            lr = args.learning_rates[idx_lr]


            if b % 10000==0:# b > 0 == 0:
                print('--- Intermediate validation... ---')
                cost, acc = test(args, model, names_valid, exp_dir, b, 
                                 current_best_cost, current_best_acc, 
                                 which_="valid", which_metric=which_val_metric)
                metric = cost if which_val_metric=='cost' else -acc
                metric_best = current_best_cost if which_val_metric=='cost' else -current_best_acc
                if metric <= metric_best:
                    model.save_model(os.path.join(exp_checkpoint_dir_best, 'model.ckpt'))
                    np.save(os.path.join(exp_checkpoint_dir_best, 'costs.npy'), np.array([*costs]))
                    np.save(os.path.join(exp_checkpoint_dir_best, 'accs.npy'), np.array([*accs]))
                    np.save(os.path.join(exp_checkpoint_dir_best, 'costs_valid.npy'), np.array([*costs_valid]))
                    np.save(os.path.join(exp_checkpoint_dir_best, 'accs_valid.npy'), np.array([*accs_valid]))
                    # test metrics. REM: another (faster) option is to reload only optimal model
                    # after training and test then...
                    test(args, model, names_test, exp_dir, b, 
                         1e20, 0., save=True, which_="test",
                         which_metric=which_val_metric)
                # update current bests
                if cost <= current_best_cost:
                    current_best_cost = cost
                if acc >= current_best_acc:
                    current_best_acc = acc

            # occasionally print stuff during training
            if b > 0 and b % 1000 == 0:
                print("average train cost over the last 500 evals: ",
                      np.mean(costs[-500:]))
                print("average valid cost over the last 100 evals: ",
                      np.mean(costs_valid[-100:]))
                print("average train acc over the last 100 evals: ", 
                      np.mean(accs[-500:]))
                print("average valid acc over the last 100 evals: ", 
                      np.mean(accs_valid[-100:]))
                print("learning rate: ", lr)

            examples, labels_target = next(b_it_train)
            numeric_in = {
                'inX': examples,
                'targetY': labels_target,
                'lr': lr,
            }
            acc_value, cost_value = model.train_model(numeric_in)
            costs += [cost_value]
            accs += [acc_value]
            
            if b % 5 == 0:
                examples, labels_target = next(b_it_valid)
                numeric_in = {
                    'inX': examples,
                    'targetY': labels_target,
                }
                _, acc_value, cost_value = model.estimate_model(numeric_in)
                costs_valid += [cost_value]
                accs_valid += [acc_value]

            if b % 5000 == 0:
                np.save(os.path.join(exp_dir, 'costs.npy'), np.array([*costs]))
                np.save(os.path.join(exp_dir, 'accs.npy'), np.array([*accs]))
                np.save(os.path.join(exp_dir, 'costs_valid.npy'), np.array([*costs_valid]))
                np.save(os.path.join(exp_dir, 'accs_valid.npy'), np.array([*accs_valid]))
                # TODO: also save (and load) current_best_cost and current_best_acc

        # also save everything after training
        model.save_model(os.path.join(exp_checkpoint_dir, 'model.ckpt'))
        np.save(os.path.join(exp_checkpoint_dir, 'costs.npy'), np.array([*costs]))
        np.save(os.path.join(exp_checkpoint_dir, 'accs.npy'), np.array([*accs]))
        np.save(os.path.join(exp_checkpoint_dir, 'costs_valid.npy'), np.array([*costs_valid]))
        np.save(os.path.join(exp_checkpoint_dir, 'accs_valid.npy'), np.array([*accs_valid]))
        
    except KeyboardInterrupt:
        print(' !!!!!!!! TRAINING INTERRUPTED !!!!!!!!')

    # finally, evaluate performance on test_set
    print('Evaluating performance on the test set...')
    _ = test(args, model, names_test, exp_dir, np.sum(args.training_batches), 
        current_best_cost)

    #write_to_comment_file(os.path.join(exp_dir, 'STATUS.txt'), 'FINISHED.')
    model.close()
    tf.reset_default_graph()
    print('###### DONE. ######')



def test(args, model, names_test, exp_dir, b_num, 
         current_best_cost=None, current_best_acc=None,
         save=True, which_="test", which_metric="cost"):

    assert which_ in ["test", "valid"]

    assert not ((current_best_cost is None) and (current_best_acc is None)), "if saving only best model, please provide current best cost metrics"
    exp_dir_best = os.path.join(exp_dir, 'best')
    if not os.path.exists(exp_dir_best):
        os.makedirs(exp_dir_best)

    # REM: we call everything 'test' but actually also use this for VALID

    try:
        names_iterator_test = shhs.patient_names_iterator(names_test)
        it_1ep_test = shhs.data_iterator_1epoch(
            names_iterator_test, #5, 1, 1, True, False)
            n_patient_queues=args.n_patient_queues, 
            epochs_before=args.eps_before,
            epochs_after=args.eps_after,
            balance_classes=args.balance_classes,
            use_if_missing_stage=args.use_if_missing_stage)
        b_it_test = shhs.batches_iterator(it_1ep_test, args.batch_size)

        costs_test = []
        accs_test = []
        labels_target_ints = []
        pred_values_ints = []

        try:
            while True:
                examples, labels_target = next(b_it_test)
                numeric_in = {
                    'inX': examples,
                    'targetY': labels_target,
                }
                pred_values_int, acc_value, cost_value = \
                model.estimate_model(numeric_in)
                costs_test += [cost_value]
                accs_test += [acc_value]

                labels_target_int = np.argmax(labels_target, axis=1)
                labels_target_ints += [labels_target_int]
                pred_values_ints += [pred_values_int]
        except StopIteration:
            pass
        finally:
            del names_iterator_test, it_1ep_test, b_it_test

        cost_test = np.mean(costs_test)
        acc_test = np.mean(accs_test)

        print('--------------------------------')
        print(which_ + ' cost: ', cost_test)
        print(which_ + ' acc:  ', acc_test)
        print('--------------------------------')

        metric_test = cost_test if which_metric=="cost" else -acc_test
        current_best_metric = current_best_cost if which_metric=="cost" else -current_best_acc

        # compare
        if (metric_test <= current_best_metric) and save:

            # labels, predictions, and counts
            labels_target_ints = np.array(labels_target_ints).flatten()
            pred_values_ints = np.array(pred_values_ints).flatten()

            print(labels_target_ints)
            print(pred_values_ints)

            # plot confmat
            confmat_test = sklearn.metrics.confusion_matrix(
                    labels_target_ints, pred_values_ints, labels=[0, 1, 2, 3, 4])
            confmat_test = np.array(confmat_test)
            utils.plot_confusion_matrix(confmat_test, classes=['W', 'S1', 'S2', 'S3&4', 'REM'], 
                save=os.path.join(exp_dir_best, which_+'_confmat.png'))

            # calculate kappa and f1-score
            kappa_test = sklearn.metrics.cohen_kappa_score(
                labels_target_ints, pred_values_ints)
            f1m_test = sklearn.metrics.f1_score(
                labels_target_ints, pred_values_ints, average='micro')
            f1M_test = sklearn.metrics.f1_score(
                labels_target_ints, pred_values_ints, average='macro')
            f1all_test = sklearn.metrics.f1_score(
                labels_target_ints, pred_values_ints, average=None)
            sk_class_rep = sklearn.metrics.classification_report(
                labels_target_ints, pred_values_ints)

            # save stuff
            np.save(os.path.join(exp_dir_best, "cost_"+which_+".npy"), cost_test)
            np.save(os.path.join(exp_dir_best, "acc_"+which_+".npy"), acc_test)
            np.save(os.path.join(exp_dir_best, "confmat_"+which_+".npy"), confmat_test)
            np.save(os.path.join(exp_dir_best, "kappa_"+which_+".npy"), kappa_test)
            np.save(os.path.join(exp_dir_best, "f1m_"+which_+".npy"), f1m_test)
            np.save(os.path.join(exp_dir_best, "f1M_"+which_+".npy"), f1M_test)
            np.save(os.path.join(exp_dir_best, "f1all_"+which_+".npy"), f1all_test)

            # write everything to a text file
            results_text = ("Batches trained at best perf: " + str(b_num) + '\n' +
                            "test cost: " + str(cost_test) + "\n" +
                            "test accuracy: " + str(acc_test) + '\n' +
                            "test kappa: " + str(kappa_test) + '\n' +
                            "test f1-micro score: " + str(f1m_test) + '\n' + 
                            "test f1-macro score: " + str(f1M_test) + '\n' + 
                            "test f1 W: " + str(f1all_test[0]) + '\n' + 
                            "test f1 S1: " + str(f1all_test[1]) + '\n' + 
                            "test f1 S2: " + str(f1all_test[2]) + '\n' + 
                            "test f1 S3&4: " + str(f1all_test[3]) + '\n' + 
                            "test f1 REM: " + str(f1all_test[4]) + '\n' +
                            "Sklearn classification report: " + sk_class_rep + '\n')
            write_to_comment_file(os.path.join(exp_dir_best, 'metrics_'+which_+'.txt'), results_text)

        return cost_test, acc_test

    except KeyboardInterrupt:
        model.close()
        tf.reset_default_graph()
        print("TEST EVAL INTERRUPTED")


if __name__ == '__main__':
    main()























