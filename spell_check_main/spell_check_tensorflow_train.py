import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMStateTuple, GRUCell
import recoverWord as rW
import json


file_name = './dic_modify_del.csv'
graph_dir = './tmp/test_logs'
save_dir = './tmp/checkpoint_dir'
word_dir = './trie.json'

class SmallConfig():
    """
    적은 학습 데이터에서의 하이퍼 파라미터
    """
    hidden_layers = 2

    #배치사이즈
    batch_size = 100
    syllable_size = 11224
    hidden_size = 256
    len_max = 7
    data_size = 9402300

    #임베딩 행렬 크기
    embedding_num = 256

    #1에폭 당 배치의 개수
    max_batches = int(data_size/batch_size)

    #배치 당 출력
    batch_print = 1000

    #에폭 수
    epoch = 20

    # 기울기 값의 상한 설정
    # 기울기 clipping을 위함
    # 기울기 L2-norm 이 max_grad_norm보다 크면 배수만큼 나누어 기울기 값을 줄인다.
    ####L2-norm :  거리
    max_grad_norm = 10

    # 학습을 할 수록 수를 줄여, learning_rate과 곱해 학습 속도를 완성한다.
    lr_decay = 0.5

config = SmallConfig()

class Seq2SeqModel():
    """Seq2Seq model usign blocks from new `tf.contrib.seq2seq`.
    Requires TF 1.0.0-alpha"""

    PAD = 0
    EOS = 0

    def __init__(self, batch_size=config.batch_size,epoch=config.epoch,
                 bidirectional=True,
                 attention=False):

        self.bidirectional = bidirectional
        self.attention = attention


        self.encoder_cell = GRUCell(config.hidden_size)
        self.decoder_cell = GRUCell(config.hidden_size*2)

        self.hidden_layers = config.hidden_layers
        self.max_batches = config.max_batches
        self.batch_print = config.batch_print
        self.max_grad_norm = config.max_grad_norm
        self.lr_decay = config.lr_decay
        self.vocab_size = config.syllable_size
        self.embedding_size = config.embedding_num
        self.batch_size = batch_size
        self.len_max = config.len_max
        self.data_size = config.data_size
        self.epoch = epoch
        self._make_graph()

    @property
    def decoder_hidden_units(self):
        return self.decoder_cell.output_size

    def _make_graph(self):
        self._init_placeholders()

        self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()

        self._init_optimizer()

    def _init_placeholders(self):
        """ Everything is time-major """

        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )
        self.decoder_targets_length = tf.placeholder(
        shape = (None,),
        dtype = tf.int32,
        name = 'decoder_targets_length',
        )
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )


    def _init_decoder_train_connectors(self):
        """
        During training, `decoder_targets`
        and decoder logits. This means that their shapes should be compatible.
        Here we do a bit of plumbing to set this up.
        """
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            #decoder_input= <EOS> + decoder_targets
            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length+1

            self.decoder_train_targets = self.decoder_targets

            #dynamic_rnn은 길이의 입력을 인자로 받기 때문에
            #모든 단어는 7의 길이로 지정해 주었다.(임시방편)
            """
            #decoder_targets의 길이를 encoder_inputs과 맞추기 위해
            #batch 내의 최대 길이를 찾아서 decoder_targets로 맞춰줌
            b_s = tf.constant(self.batch_size, dtype=tf.int64)
            self.max_targets_len = tf.stack([tf.to_int64(tf.reduce_max(self.decoder_targets_length)),b_s])
            begin = tf.constant([0,0], dtype = tf.int64)

            self.decoder_train_targets = tf.slice(self.decoder_targets, begin, self.max_targets_len)
            """
            # decoder 가중치 초기화
            with tf.name_scope('DecoderTrainFeeds'):
                self.loss_weights = tf.ones([
                    self.batch_size,
                    self.len_max
                ], dtype=tf.float32, name="loss_weights")
    def _init_embeddings(self):
        """
        음운의 embedding
        초기화 설정방법을 생각해봐야함
        """
        with tf.variable_scope("embedding") as scope:

            initializer = tf.contrib.layers.xavier_initializer()

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                initializer = initializer,
                shape=[self.vocab_size, self.embedding_size],
                dtype=tf.float32)



            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=True,
                                  dtype=tf.float32)
                )

    def _init_bidirectional_encoder(self):
        """
        input을 뒤집어서 한번 더 학습시킨다.
        """


        with tf.variable_scope("BidirectionalEncoder") as scope:

            #hidden_layer 계층을 늘린다.
            encoder_cell_fw_multi = tf.contrib.rnn.MultiRNNCell([self.encoder_cell for _ in range(self.hidden_layers)], state_is_tuple=True)
            encoder_cell_bw_multi = tf.contrib.rnn.MultiRNNCell([self.encoder_cell for _ in range(self.hidden_layers)], state_is_tuple=True)
            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw_multi,
                                                cell_bw=encoder_cell_bw_multi,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):
                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')
            else:
                encoder_fw_state = encoder_fw_state[-1]
                encoder_bw_state = encoder_bw_state[-1]

                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state),1,name='bidirectional_concat')
    def _init_decoder(self):
        """
            decoder cell.
            attention적용 시 결과가 좋지 않음.
        """
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
            decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=self.encoder_state,
                embeddings=self.embedding_matrix,
                start_of_sequence_id=self.EOS,
                end_of_sequence_id=self.EOS,
                maximum_length=self.len_max,
                num_decoder_symbols=self.vocab_size,
            )

            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=[self.len_max for _ in range(self.batch_size)],
                    time_major=True,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def _init_optimizer(self):

        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])

        #손실함수
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)

        #기울기 클리핑
        self.lr = tf.Variable(0.0, trainable=False, name='lr')

        # 훈련이 가능하다고 설정한 모든 변수들
        tvars = tf.trainable_variables()

        # 여러 값들에 대한 기울기 클리핑
        # contrib.keras.backend.gradients
        # gradients gradients of variables

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.max_grad_norm)

        #optimizer = tf.train.AdamOptimizer(self.lr)
        #self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        ###학습속도 설정
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


    def make_train_inputs(self, inputs_length_, targets_length_, inputs_, targets_ ):
        """
                feed_dict에 입력할 형태
                test 용
        """
        return {
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets_length: targets_length_,
            self.encoder_inputs: inputs_,
            self.decoder_targets: targets_,
        }

    def make_inference_inputs(self, inputs_length_, inputs_):
        """
                feed_dict에 입력할 형태
                inference 용
        """
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
        }

    def read_data(self, file_name):
        """
        오류단어길이, 목표단어길이, 오류단어, 목표단어 형식의
        csv 데이터를 읽어온다.
        단어는 각 글자를 숫자로 바꿔 저장했다.
        """

        csv_file = tf.train.string_input_producer([file_name], name='file_name')
        reader = tf.TextLineReader()
        _, line = reader.read(csv_file)
        record_defaults = [[1] for _ in range(self.len_max * 2 + 2)]
        #decode_csv는 정해진 형식(record_defaults)만 받아올 수 있기 때문에 미리 padding이 이뤄진 데이터를 준비했다.
        data = tf.decode_csv(line, record_defaults=record_defaults, field_delim=',')

        #각 데이터를 분리한다.
        #slice(분할할 데이터, 시작위치, 사이즈)
        len_error = tf.slice(data, [0], [1])
        len_target = tf.slice(data, [1], [1])
        error = tf.slice(data, [2], [self.len_max])
        target = tf.slice(data, [2 + self.len_max], [self.len_max])

        return len_error, len_target, error, target

    def read_data_batch(self,tensors):
        """
            배치로 나눠 반환한다.
        """
        len_x, len_y, x, y = tensors

        #session 단계에서 queue를 생성해줘야 한다.
        #무작위로 batch를 적용
        batch_len_x, batch_len_y, batch_x, batch_y = tf.train.shuffle_batch([len_x,len_y,x,y],
                                                                            batch_size = self.batch_size,
                                                                            capacity=30000,min_after_dequeue=3000)

        batch_len_x = tf.reshape(batch_len_x,[-1])
        batch_len_y = tf.reshape(batch_len_y,[-1])
        batch_x = tf.transpose(batch_x)
        batch_y = tf.transpose(batch_y)

        return batch_len_x, batch_len_y, batch_x, batch_y



def train_on_copy_task_(session, model,
                        len_x,len_y,x,y,
                        initial_step = 0,
                       verbose=True):
    """
            학습을 실행하는 함수
    """
    loss_track = []
    for epoch in range(initial_step,model.epoch):
        accur_epoch = 0
        loss_all = 0
        for batch in range(model.max_batches):
            all_accuracy = 0

            b_len_x, b_len_y, b_x, b_y = session.run([len_x, len_y, x, y])

            fd = model.make_train_inputs(b_len_x, b_len_y, b_x, b_y)
            _, l = session.run([model.train_op, model.loss], fd)
            if verbose:
                if batch == 0 or batch % model.batch_print == 0:
                    #그래프 출력
                    session.run(tf.local_variables_initializer())
                    summary= session.run(merged, feed_dict=fd)
                    writer.add_summary(summary, (model.max_batches*epoch)+batch)

                    print('batch {}'.format(batch))
                    print('loss {}' .format(l))
                    count = 0
                    for i, (e_in, d_ot, dt_inf) in enumerate(zip(
                            fd[model.encoder_inputs].T,
                            fd[model.decoder_targets].T,
                            #session.run(model.decoder_prediction_train, fd).T,
                            session.run(model.decoder_prediction_inference, fd).T
                    )):

                        correct = tf.equal(e_in[0:len(dt_inf)],dt_inf[0:len(e_in)])
                        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
                        all_accuracy += session.run(accuracy, fd)
                        count += 1
                    # 1에폭마다 저장한다.
                    saver.save(session, save_dir + 'BATCH.ckpt', global_step=batch)
                    all_accuracy /= count
                    print("accuracy : ",all_accuracy)
            accur_epoch += all_accuracy
        accur_epoch /= model.max_batches
        print('epoch{} : '.format(epoch),accur_epoch)


        #1에폭마다 저장한다.
        saver.save(session, save_dir+'.ckpt', global_step = epoch)


        # 학습 속도 조절
        #lr_decay = config.lr_decay ** max(((epoch + 1) * model.max_batches) + batch - config.epoch, 0.0)
        #model.assign_lr(session, config.learning_rate * lr_decay)
    return loss_track


#학습용

tf.reset_default_graph()
model = Seq2SeqModel(
                         attention=True,
                         bidirectional=True)
tensors = model.read_data(file_name)
b_len_x, b_len_y, b_x, b_y = model.read_data_batch(tensors)


#tensorboard에 graph 출력을 위해
tf.summary.scalar('cost',model.loss)

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graph_dir, session.graph)

    saver = tf.train.Saver(tf.trainable_variables())
    initial_step = 0
    ckpt = tf.train.get_checkpoint_state(save_dir)

    session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    #checkpoint가 존재할 경우 변수 값을 복구한다.
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        #복구한 시작 지점
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        print("Checkpoint")
        print(initial_step)
    else:
        print("No Checkpoint")

    train_on_copy_task_(session, model,
                           b_len_x, b_len_y, b_x, b_y,
                           initial_step,
                           verbose=True)


"""


tf.reset_default_graph()
model = Seq2SeqModel(attention=False,
                         bidirectional=True)


#trie 구조의 단어장
#단어장 내에 단어가 있는 경우 검사를 하지 않는다.
word_dir = 'C:/Users/kimhyeji/PycharmProjects/tfTest/trie.json'
dict = json.load(open(word_dir))

with tf.Session() as session:

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graph_dir, session.graph)
    session.run(tf.global_variables_initializer())


    saver = tf.train.Saver(tf.trainable_variables())
    initial_step = 0
    ckpt = tf.train.get_checkpoint_state(save_dir)

     #checkpoint가 존재할 경우 변수 값을 복구한다.
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        #복구한 시작 지점
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        print("Checkpoint")
        print(initial_step)
    while(1):
        #오타 데이터 입력을 받는다.
        word = input("입력 :")
        temp = dict
        result = 1
        #단어가 존재하지 않는 경우 result = 0
        for char in word:
            if char in temp:
                temp = temp[char]
            else:
                result = 0
                break

        #단어장에 단어가 존재하지 않는 경우
        if not result:
            index_word = rW.convert_num(word)
            fd = model.make_inference_inputs([len(word)], [[i] for i in index_word])
            inf_out = session.run(model.decoder_prediction_inference, fd).T[0]
            print(rW.recover_word(inf_out))

"""