#!/usr/bin/env python
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Produces the training and dev data for --problem into --data_dir.

generator.py produces sharded and shuffled TFRecord files of tensorflow.Example
protocol buffers for a variety of datasets registered in this file.

All datasets are registered in _SUPPORTED_PROBLEM_GENERATORS. Each entry maps a
string name (selectable on the command-line with --problem) to a function that
takes 2 arguments - input_directory and mode (one of "train" or "dev") - and
yields for each training example a dictionary mapping string feature names to
lists of {string, int, float}. The generator will be run once for each mode.
"""

import random
import tempfile

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import cnchat
from tensor2tensor.data_generators import cnchat_utils

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "", "Data directory.")
flags.DEFINE_string("tmp_dir", "data/aligned",
                    "Temporary storage directory.")
flags.DEFINE_string("problem", "",
                    "The name of the problem to generate data for.")
flags.DEFINE_integer("num_shards", 10, "How many shards to use.")
flags.DEFINE_integer("max_cases", 0,
                     "Maximum number of cases to generate (unbounded if 0).")
flags.DEFINE_integer("random_seed", 429459, "Random seed to use.")

# Mapping from problems that we can generate data for to their generators.
# pylint: disable=g-long-lambda
_SUPPORTED_PROBLEM_GENERATORS = {
    "chat_tokens_32k": (
        lambda: cnchat.chat_wordpiece_token_generator(FLAGS.tmp_dir, True, 2**15),
        lambda: cnchat.chat_wordpiece_token_generator(FLAGS.tmp_dir, False, 2**15)
    ),
}

# pylint: enable=g-long-lambda

UNSHUFFLED_SUFFIX = "-unshuffled"


def set_random_seed():
  """Set the random seed from flag everywhere."""
  tf.set_random_seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.problem not in _SUPPORTED_PROBLEM_GENERATORS:
    problems_str = "\n  * ".join(sorted(_SUPPORTED_PROBLEM_GENERATORS))
    error_msg = ("You must specify one of the supported problems to "
                 "generate data for:\n  * " + problems_str + "\n")
    raise ValueError(error_msg)

  if not FLAGS.data_dir:
    FLAGS.data_dir = tempfile.gettempdir()
    tf.logging.warning("It is strongly recommended to specify --data_dir. "
                       "Data will be written to default data_dir=%s.",
                       FLAGS.data_dir)

  set_random_seed()

  training_gen, dev_gen = _SUPPORTED_PROBLEM_GENERATORS[FLAGS.problem]

  tf.logging.info("Generating training data for %s.", FLAGS.problem)
  train_output_files = cnchat_utils.generate_files(
      training_gen(), FLAGS.problem + UNSHUFFLED_SUFFIX + "-train",
      FLAGS.data_dir, FLAGS.num_shards, FLAGS.max_cases)

  tf.logging.info("Generating development data for %s.", FLAGS.problem)
  dev_output_files = cnchat_utils.generate_files(
      dev_gen(), FLAGS.problem + UNSHUFFLED_SUFFIX + "-dev", FLAGS.data_dir, 1)

  tf.logging.info("Shuffling data...")
  for fname in train_output_files + dev_output_files:
    records = cnchat_utils.read_records(fname)
    random.shuffle(records)
    out_fname = fname.replace(UNSHUFFLED_SUFFIX, "")
    cnchat_utils.write_records(records, out_fname)
    tf.gfile.Remove(fname)


if __name__ == "__main__":
  tf.app.run()
