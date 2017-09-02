import tensorflow as tf

import sys
sys.path.append('/home/arnaud/data_these/tensorflow1/tflib/')

import ops

import convsize

class CNN():
    def __init__(self, batch_size=128, 
                 featuremap_sizes=[16, 32, 64, 128, 256], strides=[2, 2, 2, 2, 2],
                 filter_sizes=[7, 7, 7, 7, 7], hiddenlayer_size=100, 
                 balance_sm = [1, 1, 1, 1, 1], 
                 balance_cost = False, 
                 conv_type="std", batch_norm=False, activations=ops.lrelu,
                 eps_before=9, eps_after=0.5, 
                 filter=True):

        assert len(featuremap_sizes) == len(strides)
        assert len(featuremap_sizes) == len(filter_sizes)
        assert conv_type in ["std", "sep"]

        assert not (balance_sm!=[1, 1, 1, 1, 1] and balance_cost==True)

        # BUILD GRAPH
        insize_per_ep = 3750 if not filter else 1875
        insize_tot = int(insize_per_ep * (eps_before+1+eps_after))

        self.inX = tf.placeholder(
            tf.float32, 
            [batch_size, int((eps_before + 1 + eps_after)*insize_per_ep)])
        self.targetY = tf.placeholder(tf.float32, [batch_size, 5])
        self.lr = tf.placeholder(tf.float32, [])
        self.phase = tf.placeholder(tf.bool) # for batchnorm
        self.cl_viz = tf.placeholder(tf.int32, []) # class number for activation maximization (for viz)

        # get batch_size
        bs = self.inX.get_shape()[0]

        # step counter to give to saver, incremented in g opt
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # architecture
        # reshape to format adapted to conv2d
        incnn = tf.reshape(
            self.inX, [-1, 1, int(insize_per_ep*(eps_before + 1 + eps_after)), 1])
        c = incnn
        conv_outlen = insize_tot
        # apply convolutions
        if conv_type == "std":
            for l in range(len(featuremap_sizes)):
                c = tf.contrib.layers.conv2d(
                    c, featuremap_sizes[l], reuse=False,
                    kernel_size=[1, filter_sizes[l]], stride=[1, strides[l]],
                    activation_fn=None, scope="cnn"+str(l))
                if batch_norm:
                    c = tf.contrib.layers.batch_norm(c, is_training=self.phase)
                c = activations(c)
                conv_outlen = convsize.get_conv_outputsize(
                    conv_outlen, 1, filter_sizes[l], strides[l], 'same')
        elif conv_type == "sep":
            c = tf.contrib.layers.separable_conv2d(
                c, featuremap_sizes[0],
                kernel_size=[1, filter_sizes[0]],
                depth_multiplier=featuremap_sizes[1],
                stride=strides[0], reuse=False, 
                activation_fn=None, scope="cnn0")
            # separable_conv2d does not support strides of format [1, k]
            # so we do an avg pooling afterwards:
            # c = tf.nn.avg_pool(c, ksize=[1, 1, filter_sizes[0], 1],
            #     strides=[1, 1, strides[0], 1], padding='SAME')
            if batch_norm:
                c = tf.contrib.layers.batch_norm(c, is_training=phase)
            c = activations(c)
            conv_outlen = convsize.get_conv_outputsize(
                conv_outlen, 1, filter_sizes[0], strides[0], 'same')

            for l in range(1, len(featuremap_sizes)):
                c = tf.contrib.layers.separable_conv2d(
                    c, featuremap_sizes[l], 
                    kernel_size=[1, filter_sizes[l]],
                    depth_multiplier=1,
                    stride=strides[l], reuse=False, 
                    activation_fn=activations, scope="cnn"+str(l))
                # c = tf.nn.avg_pool(c, ksize=[1, 1, filter_sizes[l], 1],  
                #     strides=[1, 1, strides[l], 1], padding='SAME')
                if batch_norm:
                    c = tf.contrib.layers.batch_norm(c, is_training=phase)
                c = activations(c)
                conv_outlen = convsize.get_conv_outputsize(
                    conv_outlen, 1, filter_sizes[l], strides[l], 'same')
        else:
            raise ValueError("conv_type must be std or sep")

        outcnn = tf.squeeze(c)#c4)

        print(outcnn.get_shape())

        outcnn = tf.reshape(outcnn, [batch_size, conv_outlen*featuremap_sizes[-1]])
        print('outcnn reshaped: ', outcnn.get_shape())

        fc = ops.dense(outcnn, num_units=hiddenlayer_size, reuse=False, # TRY 200 or more..
                       nonlinearity=activations, scope="fc0")

        # a last fc layer to the softmax
        #self.outfc = ops.dense(outrnn, num_units=5, reuse=False,

        if balance_sm != [1, 1, 1, 1, 1]:
            # approximate proportions. Order: w, s1, s2, s3&4, rem
            sm_units = [0] + balance_sm#[0, 90, 10, 120, 40, 40]
            #sm_units = [0, 1, 1, 1, 1, 1]
            sm_tot_units = sum(sm_units)
            outfc = ops.dense(fc, num_units=sm_tot_units, reuse=False,
                nonlinearity=None, scope='fc_sm')
            outsm = tf.nn.softmax(outfc, dim=-1)
            preds = []
            for r in range(1, len(sm_units)):
                this_pred = tf.slice(outsm, begin=[0, sum(sm_units[:r])], size=[-1, sm_units[r]])
                print(this_pred.get_shape())
                preds += [tf.reduce_sum(this_pred, axis=-1)]
            self.preds = tf.stack(preds, axis=1)
            print(self.preds.get_shape())
            self.COST = tf.reduce_mean(
                tf.multiply(self.targetY, -tf.log(1e-8 + self.preds)))
        if balance_cost==True:
            sm_tot_units = 5
            outfc = ops.dense(fc, num_units=sm_tot_units, reuse=False,
                nonlinearity=None, scope='fc_sm')
            self.preds = tf.nn.softmax(outfc, dim=-1)
            class_proportions = [8, 1, 10, 4, 4] #approximate... refine later...
            class_weights = [1./p for p in class_proportions]
            class_weights = [w / sum(class_weights) for w in class_weights]
            class_weights = tf.constant(class_weights, tf.float32, shape=[1, 5])
            weighted_targetY = tf.multiply(self.targetY, class_weights)
            self.COST = tf.reduce_mean(
                tf.multiply(self.targetY, -tf.log(1e-8 + self.preds)))
        else:
            sm_tot_units = 5
            outfc = ops.dense(fc, num_units=sm_tot_units, reuse=False,
                nonlinearity=None, scope='fc_sm')
            self.COST = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.targetY,
                    logits=outfc))
            # for external eval
            self.preds = tf.nn.softmax(outfc, dim=-1)
        
        # predictions (for exteral eval)
        self.preds_int = tf.argmax(self.preds, axis=1)
        self.preds_true = tf.equal(
            self.preds_int, tf.argmax(self.targetY, axis=1))
        self.acc = tf.reduce_mean(tf.cast(self.preds_true, tf.float32))

        # gradients with respect to input, for viz
        self.cost_viz = tf.reduce_sum(self.preds[:, self.cl_viz])
        self.grads_viz = tf.gradients(self.cost_viz, [self.inX])[0]

        # optimization
        # tie BN statistics update to optimizer step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)#, epsilon=0.0001)
            self.OP = self.optimizer.minimize(self.COST, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables(), 
                                    max_to_keep=4)

    def init(self):
        self.session = tf.Session()
        init_op= tf.global_variables_initializer()
        self.session.run(init_op)

    def train_model(self, numeric_in):
        _, acc_value, cost_value = self.session.run(
            [self.OP, self.acc, self.COST], 
            feed_dict={
                self.inX: numeric_in['inX'],
                self.targetY: numeric_in['targetY'],
                self.lr: numeric_in['lr'],
                self.phase: 1,
            })
        return acc_value, cost_value

    def estimate_model(self, numeric_in):
        pred_values, acc_value, cost_value = self.session.run(
            [self.preds_int, self.acc, self.COST],
            feed_dict={
                self.inX: numeric_in['inX'],
                self.targetY: numeric_in['targetY'],
                self.phase: 0,
            })
        return pred_values, acc_value, cost_value

    def save_model(self, checkpoint_path):
        self.saver.save(
            self.session, checkpoint_path),
            #global_step=self.global_step)#,
            #write_meta_graph=False)
            # For some reason, I am not getting the reload to work 
            # when multiple checkpoints are kept (saving with global step)
            # (the problem is in tf.train.get_checkpoint_state)
            # So let's keep only the best model :)

    def load_model(self, checkpoint_path):
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print(len(ckpt.model_checkpoint_path))
        print(ckpt.model_checkpoint_path)
        assert len(ckpt.model_checkpoint_path) <= 255, "path is longer then 255 characters, this will not work"
        print("loading model: ", ckpt.model_checkpoint_path)
        self.saver.restore(self.session, ckpt.model_checkpoint_path)
        print('Model loading done. ')

    def get_input_grads(self, numeric_in):
        cost_value, grads_values = self.session.run(
            [self.cost_viz, self.grads_viz], feed_dict={
                self.inX: numeric_in['inX'],
                self.cl_viz: numeric_in['cl_viz'],
            })
        return cost_value, grads_values

    def close(self):
        self.session.close()








        

        




