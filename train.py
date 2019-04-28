import math
import os
import sys
import time
import numpy as np
from six.moves import xrange     # 用来兼容python 2 和 3的
import tensorflow as tf
import datautil                  # 数据预处理
import seq2seq_model


tf.reset_default_graph()
steps_per_checkpoint=200        # 两百显示一次
#从配置文件读取桶，并使用检查点序列化
#, [(20, 20)] # ["40,10","50,15"]
max_train_data_size= 0    #(0: no limit)代表输入数据的长度没有限制
dropout = 0.9 
grad_clip = 5.0
batch_size = 60
num_layers =2
learning_rate =0.5
lr_decay_factor =0.99
hidden_size = 100
checkpoint_dir= "datacn/checkpoints/"
_buckets =[(5, 5), (10, 10), (20, 20)]

def getdialogInfo():
    vocabch, rev_vocabch = datautil.initialize_vocabulary(os.path.join(datautil.data_dir, datautil.vocabulary_filech))
    vocab_sizech= len(vocabch)
    print("vocab_sizech", vocab_sizech)
    filesfrom, _ = datautil.getRawFileList(datautil.data_dir+"fromids/")
    filesto, _ = datautil.getRawFileList(datautil.data_dir+"toids/")
    source_train_file_path = filesfrom[0]
    target_train_file_path= filesto[0]
    return vocab_sizech, vocab_sizech, rev_vocabch, rev_vocabch, source_train_file_path, target_train_file_path


def main():

    vocab_sizeen, vocab_sizech, rev_vocaben, rev_vocabch, source_train_file_path, target_train_file_path = getdialogInfo()
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print("checkpoint_dir is {0}".format(checkpoint_dir))

    with tf.Session() as sess:
        model = createModel(sess, False, vocab_sizeen, vocab_sizech)
        print("Using bucket sizes:")
        print(_buckets)
        source_test_file_path = source_train_file_path
        target_test_file_path = target_train_file_path
        print(source_train_file_path)
        print(target_train_file_path)
        train_set = readData(source_train_file_path, target_train_file_path,max_train_data_size)
        test_set = readData(source_test_file_path, target_test_file_path,max_train_data_size)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        print( "bucket sizes = {0}".format(train_bucket_sizes))
        train_total_size = float(sum(train_bucket_sizes))
    
        # 桶秤是一个从0增加到1的数字列表，我们将用它来选择一个桶。[标度[i]，标度[i+1]的长度与第i个训练桶的大小成比例，稍后使用。
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        
        while True:

            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])


            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1
            

            if current_step % steps_per_checkpoint == 0:

                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                    "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),step_time, perplexity))

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                checkpoint_path = os.path.join(checkpoint_dir, "seq2seqtest.ckpt")
                print(checkpoint_path)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                for bucket_id in xrange(len(_buckets)):
                    if len(test_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)

                    _, eval_loss,output_logits = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                    
                    
                    inputstr = datautil.ids2texts(reversed([en[0] for en in encoder_inputs]) ,rev_vocaben)
                    print("输入",inputstr)
                    print("输出",datautil.ids2texts([en[0] for en in decoder_inputs] ,rev_vocabch))
  
                    outputs = [np.argmax(logit, axis=1)[0] for logit in output_logits]                    

                    if datautil.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(datautil.EOS_ID)]
                        print("结果",datautil.ids2texts(outputs,rev_vocabch))
                        
                    
                    
                sys.stdout.flush()


def createModel(session, forward_only,from_vocab_size,to_vocab_size):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
      from_vocab_size,#from
      to_vocab_size,#to
      _buckets,
      hidden_size,
      num_layers,
      dropout,
      grad_clip,
      batch_size,
      learning_rate,
      lr_decay_factor,
      forward_only=forward_only,
      dtype=tf.float32)
      
    print("model is ok")

    
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt!=None:
        model.saver.restore(session, ckpt)
        print ("Reading model parameters from {0}".format(ckpt))
    else:
        print ("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())  

    return model

def readData(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(datautil.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set



if __name__ == '__main__':
    main()
