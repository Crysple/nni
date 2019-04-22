from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import logging
import tensorflow as tf
import fcntl
import src.utils
import json_tricks
from nni.protocol import CommandType, send
import nni
from nni.tuner import Tuner
from src.utils import Logger
from src.cifar10.general_controller import GeneralController
from src.cifar10_flags import *
from collections import OrderedDict

def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


logger = build_logger("nni_controller_cifar10")


def BuildController(ControllerClass, batch_size):
    controller_model = ControllerClass(
        search_for=FLAGS.search_for,
        search_whole_channels=FLAGS.controller_search_whole_channels,
        skip_target=FLAGS.controller_skip_target,
        skip_weight=FLAGS.controller_skip_weight,
        num_cells=FLAGS.child_num_cells,
        num_layers=FLAGS.child_num_layers,
        num_branches=FLAGS.child_num_branches,
        out_filters=FLAGS.child_out_filters,
        lstm_size=64,
        lstm_num_layers=1,
        lstm_keep_prob=1.0,
        tanh_constant=FLAGS.controller_tanh_constant,
        op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
        temperature=FLAGS.controller_temperature,
        lr_init=FLAGS.controller_lr,
        lr_dec_start=0,
        lr_dec_every=1000000,  # never decrease learning rate
        l2_reg=FLAGS.controller_l2_reg,
        entropy_weight=FLAGS.controller_entropy_weight,
        bl_dec=FLAGS.controller_bl_dec,
        use_critic=FLAGS.controller_use_critic,
        optim_algo="adam",
        sync_replicas=FLAGS.controller_sync_replicas,
        num_aggregate=FLAGS.controller_num_aggregate,
        num_replicas=FLAGS.controller_num_replicas,
        batch_size=batch_size)

    return controller_model


def get_controller_ops(controller_model):
    """
    Args:
      images: dict with keys {"train", "valid", "test"}.
      labels: dict with keys {"train", "valid", "test"}.
    """

    controller_ops = {
        "train_step": controller_model.train_step,
        "train_op": controller_model.train_op,
        "lr": controller_model.lr,
        "optimizer": controller_model.optimizer
    }

    return controller_ops


class ENASTuner(Tuner):

    def __init__(self, batch_size):
        # branches defaults to 6, need to be modified according to ss
        macro_init()

        # self.child_totalsteps = (FLAGS.train_data_size + FLAGS.batch_size - 1) // FLAGS.batch_size
        #self.controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
        self.total_steps = batch_size
        logger.debug("batch_size:\t"+str(batch_size))

        ControllerClass = GeneralController
        self.controller_model = BuildController(ControllerClass, self.total_steps)

        self.graph = tf.Graph()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        self.controller_model.build_trainer()
        self.controller_ops = get_controller_ops(self.controller_model)

        hooks = []
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
            sync_replicas_hook = self.controller_ops["optimizer"].make_session_run_hook(True)
            hooks.append(sync_replicas_hook)

        self.sess = tf.train.SingularMonitoredSession(
            config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir)
        logger.debug('initlize controller_model done.')

        self.epoch = 0
        self.baseline = 0
        self.bl_dec = FLAGS.controller_bl_dec

    def generate_parameters(self, parameter_id, trial_job_id=None, pos=None):
        config = {
            'tf_variables': dict(),
            'base_line': self.baseline
        }
        values = self.sess.run(self.controller_model.tf_variables)
        for variable, value in zip(self.controller_model.tf_variables, values):
            config['tf_variables'][variable.name] = value

        return config 


    def controller_one_step(self, epoch, result):
        '''
        result:{
            default: the average of submodel's accuracy
            grads: aggregated gradients of every variable, in ascending order.
            accs: [model1's acc, model2's acc, ... modeln's acc]
        }
        '''
        grads, accs = result['grads'], result['accs']
        # Update baseline
        for acc in accs:
            self.baseline -= (1 - self.bl_dec) * (self.baseline - acc)
        # apply bp using returned gradient
        run_ops = [
            #self.controller_ops["loss"],
            #self.controller_ops["entropy"],
            self.controller_ops["lr"],
            #self.controller_ops["grad_norm"],
            #self.controller_ops["valid_acc"],
            #self.controller_ops["baseline"],
            #self.controller_ops["skip_rate"],
            self.controller_ops["train_op"]
        ]

        lr, _ = self.sess.run(run_ops, feed_dict={
            self.controller_model.grad_placeholder: grads})

        controller_step = self.sess.run(self.controller_ops["train_step"])

        log_string = "ctrl_step={:<6d}".format(controller_step)
        log_string += " lr={:<6.4f}".format(lr)
        log_string += " accs={}".format(accs)
        logger.debug(log_string)
        return


    def receive_trial_result(self, parameter_id, parameters, result):
        logger.debug("epoch:\t"+str(self.epoch))
        self.controller_one_step(self.epoch, result)

    def update_search_space(self, data):
        # Extract choice
        self.key = list(filter(lambda k: k.strip().endswith('choice'), list(data)))[0]
        data.pop(self.key)
        # Sort layers
        self.search_space = OrderedDict(sorted(data.items(), key=lambda tp:int(tp[0].split('_')[1])))
        logger.debug(self.search_space)

if __name__ == "__main__":
    tf.app.run()
