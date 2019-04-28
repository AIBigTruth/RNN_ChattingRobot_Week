from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import datautil as data_utils


class Seq2SeqModel(object):
 """------------------------------------------------------------
  带有注意力机制并且具有multiple buckets的sequence to sequence模型，
  这个类实现了一个多层循环网络组成的编码器和一个具有注意力机制的解码器。
  这个类除了使用LSTM cells还可以使用GRU cells,还使用了sampled softmax
  来处理大词汇量的输出。
  sampled softmax：设定一个词频阈值，遍历训练语料，当词表大小达到时，
  将遍历过的语料设为子集。然后清空词表，继续遍历，直到所有句子都被分入子集。
  训练模型时，在每个子集依次进行训练，只使用子集词表进行softmax。
  通俗地讲，就是将语料有策略地分成多份，在训练中使用每一份的小词表代替完整词表。
  multiple buckets：对于sequence先做聚类，预设几个固定的长度bucket，
  然后每个sequence都放到它所属的bucket里面去，然后pad到固定的长度。
  这样一来，首先我们不需要折腾while loop了，每一个bucket都是一个固定的computation graph；
  其次，每一个sequence的pad都不是很多，对于计算资源的浪费很小；
  再次，这样的实现很简单，就是一个给长度聚类，对于framework的要求很低。
  --------------------------------------------------------------"""

 def __init__(self,
               source_vocab_size,         # 原词汇的大小
               target_vocab_size,         # 目标词汇的大小
               buckets,                   # 一个（I,O）的List,I代表输入的最大长度，O代表输出的最大长度，如：[(2,4),(8,16)]
               size,                      # 模型中没层的units个数
               num_layers,                # 模型的层数
               dropout_keep_prob,         # dropout值
               max_gradient_norm,         # 截断梯度的阈值
               batch_size,                # 训练中的批次数据大小
               learning_rate,             # 开始学习率
               learning_rate_decay_factor, # 退化学习率的衰减参数
               use_lstm=False,            # 如果true，使用LSTM cells 代替GRU cells
               num_samples=512,           # sampled softmax 的样本个数
               forward_only=False,        # 如果设置了，模型只有正向传播
               dtype=tf.float32):         # internal variables的类型

    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.dropout_keep_prob_output = dropout_keep_prob
    self.dropout_keep_prob_input = dropout_keep_prob
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    #如果使用 sampled softmax,需要一个输出的映射
    output_projection = None
    softmax_loss_function = None
    # 当采样数小于vocabulary size 时，Sampled softmax才有意义
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)
      """ 函数1:自定义损失函数，计算在分类target_vocab_size里模型输出的logits与标签labels（sequence框架中的输出）之间的交叉熵                   """
      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=num_samples,
                num_classes=self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss

    """函数2：定义seq2seq框架结构，使用词嵌入量（embedding）作为输入"""
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):

      with tf.variable_scope("lstm") as scope:
          cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(size),
                                               input_keep_prob=self.dropout_keep_prob_input,
                                               output_keep_prob=self.dropout_keep_prob_output)
          if num_layers > 1:
              cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

      print("new a cell")
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)

    # 注入数据
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],name="weight{0}".format(i)))
    # 将解码器移动一位得到targets
    targets = [self.decoder_inputs[i + 1]  for i in xrange(len(self.decoder_inputs) - 1)]

    # 训练的输出和loss定义
    if forward_only:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)

      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # 梯度下降更新操作
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
               bucket_id, forward_only):

        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
          raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
          raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
          raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(target_weights), decoder_size))

        input_feed = {}
        for l in xrange(encoder_size):
          input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
          input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
          input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        if not forward_only:
          output_feed = [self.updates[bucket_id],
                         self.gradient_norms[bucket_id],
                         self.losses[bucket_id]]
        else:
          output_feed = [self.losses[bucket_id]]
          for l in xrange(decoder_size):
            output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
          return outputs[1], outputs[2], None
        else:
          return None, outputs[0], outputs[1:]


    def get_batch(self, data, bucket_id):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []


        for _ in xrange(self.batch_size):
          encoder_input, decoder_input = random.choice(data[bucket_id])

          encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
          encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
          decoder_pad_size = decoder_size - len(decoder_input) - 1
          decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                [data_utils.PAD_ID] * decoder_pad_size)
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        for length_idx in xrange(encoder_size):
          batch_encoder_inputs.append(
              np.array([encoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        for length_idx in xrange(decoder_size):
          batch_decoder_inputs.append(
              np.array([decoder_inputs[batch_idx][length_idx]
                        for batch_idx in xrange(self.batch_size)], dtype=np.int32))


          batch_weight = np.ones(self.batch_size, dtype=np.float32)
          for batch_idx in xrange(self.batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
              target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
              batch_weight[batch_idx] = 0.0
          batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
