#coding=utf-8

import os
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from tensor2tensor.data_generators.tokenizer import Tokenizer

import tensorflow as tf
import gzip
import six


_DATA_FILE_URLS = [
    # chat
    [
        "http://x/none.txt",  # pylint: disable=line-too-long
        [
            "multi100w_train_src.txt",
            "multi100w_train_target.txt"
        ]
    ],
    # diagnosis
    [
        "http://x/none.txt",  # pylint: disable=line-too-long
        [
            "question_train_src.txt",
            "question_train_target.txt"
        ]
    ],
]




def get_or_generate_vocab(tmp_dir, vocab_filename, vocab_size):
  """Generate a vocabulary from the datasets listed in _DATA_FILE_URLS."""
  vocab_filepath = os.path.join(tmp_dir, vocab_filename)
  if os.path.exists(vocab_filepath):
    vocab = SubwordTextEncoder(vocab_filepath)
    return vocab

  tokenizer = Tokenizer()
  for source in _DATA_FILE_URLS:
    for lang_file in source[1]:
      tf.logging.info("Reading file: %s" % lang_file)
      filepath = os.path.join(tmp_dir, lang_file)


      # Use Tokenizer to count the word occurrences.
      with tf.gfile.GFile(filepath, mode="r") as source_file:
        for line in source_file:
          line = line.strip()
          _ = tokenizer.encode(line)

  vocab = SubwordTextEncoder.build_to_target_size(
      vocab_size, tokenizer.token_counts, vocab_filepath, 1, 1e3)
  return vocab


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s", str((k, v)))
    if isinstance(v[0], six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value is neither an int nor a float; v: %s type: %s" %
                       (str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))



def generate_files(generator,
                   output_name,
                   output_dir,
                   num_shards=1,
                   max_cases=None):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_name: the file name prefix under which output will be saved.
    output_dir: directory to save the output to.
    num_shards: how many shards to use (defaults to 1).
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.

  Returns:
    List of output file paths.
  """
  writers = []
  output_files = []
  for shard in xrange(num_shards):
    output_filename = "%s-%.5d-of-%.5d" % (output_name, shard, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    output_files.append(output_file)
    writers.append(tf.python_io.TFRecordWriter(output_file))

  counter, shard = 0, 0
  for case in generator:
    if counter % 100000 == 0:
      tf.logging.info("Generating case %d for %s." % (counter, output_name))
    counter += 1
    if max_cases and counter > max_cases:
      break
    sequence_example = to_example(case)
    writers[shard].write(sequence_example.SerializeToString())
    shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()

  return output_files


def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 10000 == 0:
      tf.logging.info("read: %d", len(records))
  return records


def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
    if count % 10000 == 0:
      tf.logging.info("write: %d", count)
  writer.close()