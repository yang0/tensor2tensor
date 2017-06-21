import generator_utils
import os
import tensorflow as tf

_TRAIN_DATASETS = [
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

_TEST_DATASETS = [
    # chat
    [
        "http://x/none.txt",  # pylint: disable=line-too-long
        [
            "test_src.txt",
            "test_target.txt"
        ]
    ],
]


def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  lang1_lines, lang2_lines = [], []
  for dataset in datasets:
    url = dataset[0]
    compressed_filename = os.path.basename(url)
    compressed_filepath = os.path.join(tmp_dir, compressed_filename)

    lang1_filename, lang2_filename = dataset[1]
    lang1_filepath = os.path.join(tmp_dir, lang1_filename)
    lang2_filepath = os.path.join(tmp_dir, lang2_filename)


    with tf.gfile.GFile(lang1_filepath, mode="r") as lang1_file:
      with tf.gfile.GFile(lang2_filepath, mode="r") as lang2_file:
        lang1_file_lines = lang1_file.readlines()
        lang2_file_lines = lang2_file.readlines()
        assert len(lang1_file_lines) == len(lang2_file_lines), lang1_filepath
        lang1_lines.extend(lang1_file_lines)
        lang2_lines.extend(lang2_file_lines)

  write_chunk_size = 10000
  assert len(lang1_lines) == len(lang2_lines)
  with tf.gfile.GFile(filename + ".lang1", mode="w") as lang1_file:
    i = 0
    while i <= len(lang1_lines):
      for line in lang1_lines[i * write_chunk_size:(i + 1) * write_chunk_size]:
        lang1_file.write(line)
      i += 1
    for line in lang1_lines[i * write_chunk_size:]:
      lang1_file.write(line)
  with tf.gfile.GFile(filename + ".lang2", mode="w") as lang2_file:
    i = 0
    while i <= len(lang2_lines):
      for line in lang2_lines[i * write_chunk_size:(i + 1) * write_chunk_size]:
        lang2_file.write(line)
      i += 1
    for line in lang2_lines[i * write_chunk_size:]:
      lang2_file.write(line)
  return filename


def token_generator(source_path, target_path, token_vocab, eos=None):
  """Generator for sequence-to-sequence tasks that uses tokens.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are token ids from the " "-split source (and target, resp.) lines
  converted to integers using the token_map.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ints = token_vocab.encode(source.strip()) + eos_list
        target_ints = token_vocab.encode(target.strip()) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()

def ende_wordpiece_token_generator(tmp_dir, train, vocab_size):
  symbolizer_vocab = generator_utils.get_or_generate_vocab(
      tmp_dir, "tokens.vocab.%d" % vocab_size, vocab_size)
  datasets = _TRAIN_DATASETS if train else _TEST_DATASETS
  tag = "train" if train else "dev"

  data_path = _compile_data(tmp_dir, datasets, "chinese_chat_tok_%s" % tag)
  return token_generator(data_path + ".lang1", data_path + ".lang2",
                         symbolizer_vocab, 1)