from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import collections
import csv
import os
import optimization
import network
import bert
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("NN_hidden_size", None, "The hidden size of encoder CNN or PCNN")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_float(
    "margin", 1.0,
    "The max position embedding length"
)

flags.DEFINE_bool("use_pcnn", False, "Whether to use PCNN")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, head, tail, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.head = head
        self.tail = tail
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 head_ids,
                 tail_ids,
                 position1_ids,
                 position2_ids,
                 segment_mask,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.head_ids = head_ids
        self.tail_ids = tail_ids
        self.position1_ids = position1_ids
        self.position2_ids = position2_ids
        self.segment_mask = segment_mask
        self.label_id = label_id
        self.is_real_example = is_real_example


class Data_loader():
    def __init__(self):
        self.labels = []
        with open(FLAGS.data_dir + "rel2id.txt", 'r') as f:
            lines = f.readlines()
        for rel in lines:
            self.labels.append(rel.strip())

    def get_train_example(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, 'train.csv')), "train")

    def get_eval_example(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, 'eval.csv')), "eval")

    def get_test_example(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, 'test.csv')), "test")

    def get_labels(self):
        return self.labels

    def _read_csv(cls, input_file):
        with open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[3])
            head = tokenization.convert_to_unicode(line[1])
            tail = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text=text, label=label, head=head, tail=tail)
            )
        return examples


def model_fn_builder(
        bert_config,
        num_labels,
        init_checkpoint,
        learning_rate,
        num_train_steps,
        num_warmup_steps,
        use_tpu,
        use_one_hot_embeddings):
    def model_fn(features, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        head_ids = features["head_ids"]
        tail_ids = features["tail_ids"]
        position1_ids = features["position1_ids"]
        position2_ids = features["position2_ids"]
        segment_mask = features["segment_mask"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_exmaple" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (per_loss, total_loss, logits, probabilities) = create_model(bert_config, is_training, FLAGS.use_pcnn,
                                                                     input_ids, input_mask, head_ids, tail_ids,
                                                                     num_labels, use_one_hot_embeddings,
                                                                     segment_mask, position1_ids, position2_ids)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assigment_map, initialized_variable_names) = bert.get_assignment_map_from_checkpoint(tvars,
                                                                                                  init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assigment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in (tvars):
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,use_tpu)

            output_spec = tf.contrib.tpu.TRUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn
            )
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example
                )
                precision = tf.metrics.precision(
                    labels=label_ids, predictions=predictions, weights=is_real_example
                )
                recall = tf.metrics.recall(
                    labels=label_ids, predictions=predictions, weights=is_real_example
                )
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_loss": loss
                }

            eval_metrics = (metric_fn, [per_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TRUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn
            )
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def create_model(bert_config, is_training, use_pcnn,
                 input_ids, input_mask, head_ids, tail_ids, num_labels, use_one_hot_embeddings,
                 segment_mask, position1, position2):
    model = bert.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        head_ids=head_ids,
        tail_ids=tail_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    # output_layer = tf.concat(output_layer, tf.reshape(position1,[position1.shape[0],FLAGS.max_seq_length,1]), axis=1)
    # output_layer = tf.concat(output_layer, tf.reshape(position2,[position2.shape[0],FLAGS.max_seq_length,1]), axis=1)
    # word_embedding_size = output_layer.shape[-1].value

    head_embedding = model.get_head_embedding()
    tail_embedding = model.get_tail_embedding()

    # neg_head_embedding = tf.random_shuffle(head_embedding)
    # neg_tail_embedding = tf.random_shuffle(tail_embedding)

    # [batch_size, hidden_size]
    sentence_embedding = tf.layers.conv1d(inputs=output_layer,
                                          filters=bert_config.hidden_size,
                                          kernel_size=3,
                                          strides=1,
                                          padding="same",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    if use_pcnn:
        mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        mask = tf.nn.embedding_lookup(mask_embedding,segment_mask)
        sentence_embedding = tf.reduce_max(tf.expand_dims(mask*100,2)+tf.expand_dims(sentence_embedding,3),axis=1) - 100
        return tf.reshape(sentence_embedding,[-1,bert_config.hidden_size*3])
    else:
        sentence_embedding = tf.reduce_max(sentence_embedding,axis=-2)
    # if use_pcnn:
    #     sentence_embedding = network.sentence_encoder(output_layer, segment_mask, bert_config.hidden_size, True)
    # else:
    #     sentence_embedding = network.sentence_encoder(output_layer, segment_mask, bert_config.hidden_size, False)



    output_weights = tf.get_variable("output_weights", [num_labels,bert_config.hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [num_labels],
                                  initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            sentence_embedding = tf.nn.dropout(sentence_embedding, keep_prob=0.9)
        pos = tf.add(head_embedding,tail_embedding)
        pos = abs(tf.add(pos,-sentence_embedding))
        pos = tf.reduce_sum(pos, axis=1, keep_dims=True)
        # neg = tf.reduce_sum(abs(neg_head_embedding + tail_embedding - sentence_embedding), axis=1, keep_dims=True)

        # per_trans_loss = tf.maximum(pos - neg + FLAGS.margin, 0)
        per_trans_loss = tf.maximum(pos + FLAGS.margin, 0)
        total_trans_loss = tf.reduce_mean(per_trans_loss)

        logits = tf.matmul(sentence_embedding, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits)

        return per_trans_loss, total_trans_loss, logits, probabilities

    return per_trans_loss, total_trans_loss, sentence_embedding


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_idx, example) in enumerate(examples):
        if ex_idx % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_idx, len(examples)))

        feature = convert_single_example(ex_idx, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["head_ids"] = create_int_feature([feature.head_ids])
        features["tail_ids"] = create_int_feature([feature.tail_ids])
        features['position1_ids'] = create_int_feature(feature.position1_ids)
        features['position2_ids'] = create_int_feature(feature.position2_ids)
        features['segment_mask'] = create_int_feature(feature.segment_mask)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label.strip()] = i

    tokens = tokenizer.tokenize(example.text)
    head = example.head
    tail = example.tail
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    final_tokens = []
    final_tokens.append("[CLS]")
    for token in tokens:
        final_tokens.append(token)
    final_tokens.append("[SEP]")

    input_ids = tokenizer.convert_tokens_to_ids(final_tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    def get_entity_pos(entity):
        ids = 0
        entity = entity.split(" ")
        for i, word in enumerate(final_tokens):
            if word == entity[0]:
                ids = i
                break
        return ids

    head_ids = get_entity_pos(head)
    tail_ids = get_entity_pos(tail)

    segment_mask = []
    position1_ids = []
    position2_ids = []
    for i in range(len(input_ids)):
        position1_ids.append(i - head_ids + max_seq_length)
        position2_ids.append(i - tail_ids + max_seq_length)
        if i <= min(head_ids, tail_ids):
            segment_mask.append(0)
        elif i < max(head_ids, tail_ids):
            segment_mask.append(1)
        else:
            segment_mask.append(2)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(position1_ids) == max_seq_length
    assert len(position2_ids) == max_seq_length
    assert len(segment_mask) == max_seq_length

    if example.label.strip() in label_map.keys():
        label_id = label_map[example.label.strip()]
    else:
        label_id = label_map["NA"]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join((str(x) for x in input_mask)))
        tf.logging.info("position1_ids: %s" % " ".join([str(x) for x in position1_ids]))
        tf.logging.info("position2_ids: %s" % " ".join([str(x) for x in position2_ids]))
        tf.logging.info("segment_mask: %s" % " ".join(([str(x) for x in segment_mask])))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        head_ids=head_ids,
        tail_ids=tail_ids,
        position1_ids=position1_ids,
        position2_ids=position2_ids,
        segment_mask=segment_mask,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "head_ids": tf.FixedLenFeature([1], tf.int64),
        "tail_ids": tf.FixedLenFeature([1], tf.int64),
        "position1_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "position2_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder
            )
        )
        return d

    return input_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = bert.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    data_loader = Data_loader()

    label_list = data_loader.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = data_loader.get_train_example(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.data_dir, "train.tf_record")
        # file_based_convert_examples_to_features(
        #     train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = data_loader.get_eval_example(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(())
        eval_file = os.path.join(FLAGS.data_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info(" Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples, len(eval_examples) - num_actual_eval_examples)
        tf.logging.info(" Batch size = %d", FLAGS.eval_batch_size)

        eval_step = None
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_step = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_step)

        output_eval_file = os.path.join(FLAGS.out_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
            predict_examples = data_loader.get_test_example(FLAGS.data_dir)
            num_actual_predict_examples = len(predict_examples)
            predict_file = os.path.join(FLAGS.data_dir, "predict.tf_record")
            file_based_convert_examples_to_features(predict_examples, label_list,
                                                    FLAGS.max_seq_length, tokenizer,
                                                    predict_file)
            tf.logging.info("***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(predict_examples), num_actual_predict_examples,
                            len(predict_examples) - num_actual_predict_examples)
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

            predict_drop_remainder = True if FLAGS.use_tpu else False
            predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder
            )

            result = estimator.predict(input_fn=predict_input_fn)

            output_predict_file = os.path.join(FLAGS.output_dir, "test_results.csv")
            with tf.gfile.GFile(output_predict_file, 'w') as writer:
                num_written_lines = 0
                tf.logging.info("***** Predict results *****")
                for (i, prediction) in enumerate(result):
                    probabilities = prediction["probabilities"]
                    if i >= num_actual_predict_examples:
                        break
                    output_line = "\t".join(
                        str(class_probability) for class_probability in probabilities) + "\n"
                    writer.write(output_line)
                    num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
