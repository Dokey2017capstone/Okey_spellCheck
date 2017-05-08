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
    len_max = 7
    data_size = 35228


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
        self.len_max = config.len_max
        self.data_size = config.data_size

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

        csv_file = tf.train.string_input_producer([file_name], name='file_name')
        reader = tf.TextLineReader()
        _, line = reader.read(csv_file)
        record_defaults = [[1] for _ in range(self.len_max * 2 + 2)]
        data = tf.decode_csv(line,record_defaults = record_defaults,field_delim=',')
        len_error = tf.slice(data, [0],[1])
        len_target = tf.slice(data,[1],[1])
        error = tf.slice(data,[2],[self.len_max])
        target = tf.slice(data, [2+self.len_max], [self.len_max])
        return len_error, len_target, error, target

    def read_data_batch(self,file_name):
        """
            배치로 나눠 준다.
        """

        len_x, len_y, x, y = self.read_data(file_name)
        batch_len_x, batch_len_y, batch_x, batch_y = tf.train.batch([len_x,len_y, x,y], dynamic_pad = True, batch_size = self.batch_size)
        return batch_len_x, batch_len_y, batch_x, batch_y

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

        self.encoder_input_len = batch_len
        self.encoder_input_data, self.decoder_target_data, self.decoder_target_len = batch_tensors


    def embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            embedding_size = [self.syllable_size, self.hidden_size]
            self.embedding = tf.get_variable("embedding",
                                             shape=embedding_size,
                                             initializer=initializer,
                                             dtype=tf.float32)

            #self.decoder_embedding_input = tf.nn.embedding_lookup(self.embedding, self.decoder_input_data)

    def init_encoder_cell(self):
        with tf.variable_scope("encoder_cell") as scope:
            encoder_cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
            encoder_cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_final_state,
              encoder_bw_final_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                cell_bw=encoder_cell_bw,
                                                inputs=self.encoder_embedding_input,
                                                sequence_length=self.encoder_inputs_length,
                                                dtype=tf.float32, time_major=True)
            )

            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            encoder_final_state_c = tf.concat(
                (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            encoder_final_state =tf.contirb.rnn. LSTMStateTuple(
                c=encoder_final_state_c,
                h=encoder_final_state_h
            )
    def make_train_inputs(self, x, y, len_x, len_y):
        #학습용 배치를 제작
        self.shuffle_bucket_batch(len_x, [x,y,len_y])


    def make_test_inputs(self):
        #테스트용 배치 입력 제작
        return 1

    def make_model(self):
        self.init_placeholders()
        len_x, len_y, x, y = self.read_data_batch(file_name)
        self.embeddings()

        for i in range(self.data_size/self.batch_size):
            self.make_train_inputs(x,y,len_x,len_y)
            self.encoder_embedding_input = tf.nn.embedding_lookup(self.embedding, self.encoder_input_data)



s = Seq2Seq()
s.make_model()



#queue를 시작하기 위해서는 thread를 돌려야한다.
#https://www.reddit.com/r/tensorflow/comments/5z1o7q/tensorflow_freezing_when_accessing_csv_data/
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    sess.run(tf.global_variables_initializer())

    #에폭
    for i in range(10):
        error = sess.run([s.encoder_embedding_input])
        print('test[%d]'%i)
        print(error)
