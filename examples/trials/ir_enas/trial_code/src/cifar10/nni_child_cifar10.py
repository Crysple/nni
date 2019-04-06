from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time
import fcntl
import numpy as np
np.random.seed(996)
import tensorflow as tf
tf.set_random_seed(996)
import logging
import pickle
from src.utils import Logger
from src.cifar10.data_utils import read_data
from src.cifar10.general_child import GeneralChild
from src.cifar10_flags import *
child_init()


def build_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_name+'.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


logger = build_logger("nni_child_cifar10")


def BuildChild(images, labels, ChildClass):
    child_model = ChildClass(
        images,
        labels,
        use_aux_heads=FLAGS.child_use_aux_heads,
        cutout_size=FLAGS.child_cutout_size,
        whole_channels=FLAGS.controller_search_whole_channels,
        num_layers=FLAGS.child_num_layers,
        num_cells=FLAGS.child_num_cells,
        num_branches=FLAGS.child_num_branches,
        fixed_arc=FLAGS.child_fixed_arc,
        out_filters_scale=FLAGS.child_out_filters_scale,
        out_filters=FLAGS.child_out_filters,
        keep_prob=FLAGS.child_keep_prob,
        drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
        num_epochs=FLAGS.num_epochs,
        l2_reg=FLAGS.child_l2_reg,
        data_format=FLAGS.data_format,
        batch_size=FLAGS.batch_size,
        clip_mode="norm",
        grad_bound=FLAGS.child_grad_bound,
        lr_init=FLAGS.child_lr,
        lr_dec_every=FLAGS.child_lr_dec_every,
        lr_dec_rate=FLAGS.child_lr_dec_rate,
        lr_cosine=FLAGS.child_lr_cosine,
        lr_max=FLAGS.child_lr_max,
        lr_min=FLAGS.child_lr_min,
        lr_T_0=FLAGS.child_lr_T_0,
        lr_T_mul=FLAGS.child_lr_T_mul,
        optim_algo="momentum",
        sync_replicas=FLAGS.child_sync_replicas,
        num_aggregate=FLAGS.child_num_aggregate,
        num_replicas=FLAGS.child_num_replicas,
        mode=FLAGS.child_mode
    )

    return child_model


def get_child_ops(child_model):
    child_ops = {
        "global_step": child_model.global_step,
        "loss": child_model.loss,
        "train_op": child_model.train_op,
        "lr": child_model.lr,
        "grad_norm": child_model.grad_norm,
        "train_acc": child_model.train_acc,
        "optimizer": child_model.optimizer,
        "num_train_batches": child_model.num_train_batches,
        "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
        "eval_func": child_model.eval_once,
    }
    return child_ops


class ENASTrial():

    def __init__(self):
        if FLAGS.child_mode == 'subgraph':
            images, labels = read_data(FLAGS.data_path, num_valids=0)
        else:
            images, labels = read_data(FLAGS.data_path)

        self.output_dir = os.path.join(os.getenv('NNI_OUTPUT_DIR'), '../..')
        self.file_path = os.path.join(self.output_dir, 'trainable_variable.txt')

        self.g = tf.Graph()
        with self.g.as_default():
            self.child_model = BuildChild(images, labels, GeneralChild)

            self.total_data = {}

            self.child_model.connect_controller()
            if FLAGS.child_mode != 'subgraph':
                self.child_model.build_valid_rl()
            self.child_ops = get_child_ops(self.child_model)
            config = tf.ConfigProto(
                device_count={"CPU":8},
                intra_op_parallelism_threads=0,
                inter_op_parallelism_threads=0,
                allow_soft_placement=True)

            self.sess = tf.train.SingularMonitoredSession(config=config)

        logger.debug('initlize ENASTrial done.')

    def run_child_one_macro(self):
        run_ops = [
            self.child_ops["loss"],
            self.child_ops["lr"],
            self.child_ops["grad_norm"],
            self.child_ops["train_acc"],
            self.child_ops["train_op"],
        ]
        actual_step = None
        loss, lr, gn, tr_acc, _ = self.sess.run(run_ops)
        global_step = self.sess.run(self.child_ops["global_step"])
        log_string = ""
        log_string += "ch_step={:<6d}".format(global_step)
        log_string += " loss={:<8.6f}".format(loss)
        log_string += " lr={:<8.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " tr_acc={:<3d}/{:>3d}".format(tr_acc, FLAGS.batch_size)
        if int(global_step) % FLAGS.log_every == 0:
            logger.debug(log_string)
        return loss, global_step

    def get_csvaa(self):
        cur_valid_acc = self.sess.run(self.child_model.cur_valid_acc)
        return cur_valid_acc

    def start_eval_macro(self):
        self.child_ops["eval_func"]\
            (self.sess, "valid", self.child_model)
        

    def train_on_this(self):
        max_acc = 0
        while True:
            loss, global_step = self.run_child_one_macro()
            if global_step % self.child_ops['num_train_batches'] == 0:
                acc = self.child_ops["eval_func"](self.sess, "test", self.child_model)
                max_acc = max(max_acc, acc)
                '''@nni.report_intermediate_result(acc)'''
            if global_step / self.child_ops['num_train_batches'] >= FLAGS.num_epochs:
                '''@nni.report_final_result(max_acc)'''
                break

    def run(self, num):
        for _ in range(num):
            if FLAGS.child_mode == 'subgraph':
                """@nni.get_next_parameter()"""
            else:
                """@nni.get_next_parameter(tf, self.sess)"""

            """@nni.variable(nni.choice('train', 'validate'), name=entry)"""
            entry = 'trian'
            if entry == 'train':
                loss, _ = self.run_child_one_macro()
                '''@nni.report_final_result(loss)'''
            elif entry == 'validate':
                valid_acc_arr = self.get_csvaa()
                '''@nni.report_final_result(valid_acc_arr)'''
            else:
                raise RuntimeError('No such entry: ' + entry)

def main(_):
    logger.debug("-" * 80)

    if not os.path.isdir(FLAGS.output_dir):
        logger.debug("Path {} does not exist. Creating.".format(FLAGS.output_dir))
        os.makedirs(FLAGS.output_dir)
    elif FLAGS.reset_output_dir:
        logger.debug("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
        shutil.rmtree(FLAGS.output_dir)
        os.makedirs(FLAGS.output_dir)
    logger.debug("-" * 80)
    trial = ENASTrial()
    controller_total_steps = FLAGS.controller_train_steps * FLAGS.controller_num_aggregate
    logger.debug("here is the num train batches")

    logger.debug(trial.child_model.num_train_batches)
    child_totalsteps = (FLAGS.train_data_size + FLAGS.batch_size - 1) // FLAGS.batch_size
    logger.debug("child total \t"+str(child_totalsteps))
    epoch = 0

    trial.train_on_this()

if __name__ == "__main__":
    tf.app.run()
