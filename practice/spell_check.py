#-*- coding:utf-8 -*-
import tensorflow as tf

def read_data(file_name):
    try:
        csv_file = tf.train.string_input_producer([file_name], name='file_name')
        reader = tf.TextLineReader()
        _, line = reader.read(csv_file)
        len_error, error, target = tf.decode_csv(line,record_defaults=[[0],[""],[""]],field_delim=',')
    except:
        print("read_error")
    return len_error, error, target

#배치로 나눠 준다.
def read_data_batch(file_name,batch_size):
    len_x, x, y = read_data(file_name)
    batch_len_x, batch_x, batch_y = tf.train.batch([len_x,x,y], batch_size = batch_size)
    return batch_len_x, batch_x, batch_y

#배치를 만들때, 길이가 다른 경우 패딩을 하게 되는데
#배치 내 단어의 길이 차이가 심할 수록 패딩의 양이 늘어나고
#학습에 좋지 않기 때문에 비슷한 길이의 단어끼리 배치를 만든다.
def shuffle_bucket_batch(input_len, tensors, batch_size):
    #랜덤 배치
    table_index = tf.train.range_input_producer(
        int(input_len.get_shape()[0]), shuffle=True
    ).dequeue()

    # 배치를 문자의 길이에 맞게 배치 한다.
    batch_len, batch_tensors = tf.contrib.training.bucket_by_sequence_length(
        input_length=tf.gather(input_len, table_index),
        tensors=tensors,
        batch_size=batch_size,
        # 길이 범위
        bucket_boundaries=[2, 3, 4, 6],
        dynamic_pad=True,
        # 대기열에 수용할 수 있는 텐서의 수
        capacity=32)
    return tuple(batch_tensors)

file_name = 'C:/Users/kimhyeji/PycharmProjects/tfTest/dic_modify.csv'

len_x, x, y = read_data_batch(file_name,20)
source_batch, target_batch = shuffle_bucket_batch(len_x,[x,y],20)

#queue를 시작하기 위해서는 thread를 돌려야한다.
#https://www.reddit.com/r/tensorflow/comments/5z1o7q/tensorflow_freezing_when_accessing_csv_data/
with tf.train.MonitoredSession() as sess:
    error , target = sess.run([source_batch, target_batch])