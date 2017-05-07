#-*- coding:utf-8 -*-
import tensorflow as tf
import math

file_name = 'C:/Users/kimhyeji/PycharmProjects/tfTest/dic_modify.csv'


class SmallConfig():
    """
    적은 학습 데이터에서의 하이퍼 파라미터
    """

    batch_size = 20
    syllable_size = 11224
    hidden_size = 200


config = SmallConfig()

class Seq2Seq(object):
    """
        입력: 한글 음절로 이루어진 오타 데이터
        출력 :한글 음절로 이루어진 정답 데이터
        입력과 출력은 다른 길이를 가질 수 있다.
    """
    def __init__(self):
        self.batch_size = config.batch_size
        self.syllable_size = config.syllable_size
        self.hidden_size = config.hidden_size

    def init_placeholders(self):
        self.encoder_input_data = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_input_data'
        )

        self.decoder_target_data = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_target_data'
        )

        self.encoder_input_len = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_input_length'
        )

        self.decoder_target_len = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_input_length'
        )

    def read_data(self,file_name):
        try:
            csv_file = tf.train.string_input_producer([file_name], name='file_name')
            reader = tf.TextLineReader()
            _, line = reader.read(csv_file)
            len_error, error, target = tf.decode_csv(line,record_defaults=[[0],[""],[""]],field_delim=',')
        except:
            print("read_error")
        return len_error, error, target

    def read_data_batch(self,file_name):
        """
            배치로 나눠 준다.
        """

        len_x, x, y = self.read_data(file_name)
        batch_len_x, batch_x, batch_y = tf.train.batch([len_x,x,y], batch_size = self.batch_size)
        return batch_len_x, batch_x, batch_y

    def shuffle_bucket_batch(self,input_len, tensors):
        """
            배치를 만들때, 길이가 다른 경우 패딩을 하게 되는데
            배치 내 단어의 길이 차이가 심할 수록 패딩의 양이 늘어나고
            학습에 좋지 않기 때문에 비슷한 길이의 단어끼리 배치를 만든다.
        """

        #랜덤 배치
        table_index = tf.train.range_input_producer(
            int(input_len.get_shape()[0]), shuffle=True
        ).dequeue()

        # 배치를 문자의 길이에 맞게 배치 한다.
        batch_len, batch_tensors = tf.contrib.training.bucket_by_sequence_length(
            input_length=tf.gather(input_len, table_index),
            tensors=tensors,
            batch_size=self.batch_size,
            # 길이 범위
            bucket_boundaries=[2, 3, 4, 6],
            dynamic_pad=True,
            # 대기열에 수용할 수 있는 텐서의 수
            capacity=120)
        return tuple(batch_tensors)

    def embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            embedding_size = [self.syllable_size, self.hidden_size]
            self.embedding = tf.get_variable("embedding",
                                             shape=embedding_size,
                                             initializer=initializer,
                                             dtype=tf.float32)

            self.encoder_embedding_input = tf.nn.embedding_lookup(self.embedding, self.encoder_input_data)

            #self.decoder_embedding_input = tf.nn.embedding_lookup(self.embedding, self.decoder_input_data)

    def make_model(self):
        self.init_placeholders()
        len_x, x, y = self.read_data_batch(file_name)
        self.source_batch, self.target_batch = self.shuffle_bucket_batch(len_x, [x, y])

        self.embeddings()


s = Seq2Seq()
s.make_model()



#queue를 시작하기 위해서는 thread를 돌려야한다.
#https://www.reddit.com/r/tensorflow/comments/5z1o7q/tensorflow_freezing_when_accessing_csv_data/
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for i in range(10):
        error , target = sess.run([s.source_batch, s.target_batch])
        print('test[%d]'%i)
        print(error)
