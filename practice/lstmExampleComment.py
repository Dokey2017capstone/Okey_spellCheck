import time
import numpy as np
import tensorflow as tf
import inspect
import os

from tensorflow.models.tutorials.rnn.ptb import reader

class SmallConfig():
    # 가중치 행렬 초기화 시 생성되는 값의 범위
    init_scale = 0.1

    # 경사 하강법의 학습 속도
    learning_rate = 1.0

    # 기울기 값의 상한 설정
    # 기울기 clipping을 위함
    # 기울기 L2-norm 이 max_grad_norm보다 크면 배수만큼 나누어 기울기 값을 줄인다.
    ####L2-norm :  거리
    max_grad_norm = 5

    # 순환 신경망을 구성할 계층의 개수
    num_layers = 2

    # 연속적으로 처리할 데이터 양
    # 횟수만큼 가중치 학습 후 경사 하강법으로 기울기 업데이트
    num_steps = 20

    # 한 계층에 배치할 뉴런의 수, 셀의 개수
    hidden_size = 200

    # 초기 학습 속도를 유지하는 epoch 횟수
    max_epoch = 4

    # 총 epoch 횟수
    max_max_epoch = 13

    # dropout 하지 않을 확률
    keep_prob = 1.0

    # 학습을 할 수록 수를 줄여, learning_rate과 곱해 학습 속도를 완성한다.
    lr_decay = 0.5

    # 학습 데이터의 일부만을 사용하는 mini-batch
    batch_size = 20

    # 총 단어의 개수
    vocab_size = 10000


#학습 시 사용
config = SmallConfig()

#test 시 사용
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1


# 참고로 (object)는 new style class 를 의미한다.
class PTBModel(object):
    def __init__(self, config, is_training=False):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        input_size = [config.batch_size, config.num_steps]
        self.input_data = tf.placeholder(tf.int32, input_size)
        self.targets = tf.placeholder(tf.int32, input_size)

        # hidden size의 셀 개수를 가진 LSTM cell을 구성한다.
        def lstm_cell():
            ###version에 따른 error 임시 수정. inspect 제거 필요함.
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    config.hidden_size, forget_bias=0.0, state_is_tuple=True)
        attn_cell = lstm_cell

        # 드롭아웃, 현재는 적용하지 않는 상태
        ###? 상위 계층 전달 X, 다음 셀 적용 가능
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), config.keep_prob)

        # 신경망의 계층 생성
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(SmallConfig.num_layers)], state_is_tuple=True)

        # 셀의 초기 설정
        ###? 한 셀에 입력받는다.  초기 batch_size
        self.initial_state = cell.zero_state(config.batch_size, tf.float32)

        embedding_size = [config.vocab_size, config.hidden_size]
        embedding = tf.get_variable("embedding", embedding_size)
        # params = tf.constant([10,20,30,40])
        # ids = tf.constant([1,1,3])
        # print tf.nn.embedding_lookup(params,ids).eval()
        # [20 20 40]
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 드롭아웃, 현재는 적용하지 않는 상태
        ###? 은닉상태와 셀 상태에 입력값을 넣지 않는다.
        if is_training and config.keep_prob < 1:
            inputs = tf.contrib.rnn.DropoutWrapper(inputs, config.keep_prob)

        outputs = []
        state = self.initial_state

        # 변수의 namespace
        # https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/how_tos/variable_scope/
        with tf.variable_scope("RNN"):
            # 입력 데이터 양(개수)만큼
            for time_step in range(config.num_steps):
                ###? 같은 이름의 변수를 공유하겠다.
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                # (상위 계층으로 전달되는)은닉상태, 셀 상태
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                # 20*1*200 을 num_steps만큼 더해서 20*20*200
                outputs.append(cell_output)

        # logits 계산을 위해 output을 20*20*200 -> 400*200
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

        ###? hidden node를 계산한다.
        # 200*10000
        softmax_w_size = [config.hidden_size, config.vocab_size]
        softmax_w = tf.get_variable("softmax_w", softmax_w_size)
        softmax_b = tf.get_variable("softmax_b", [config.vocab_size])

        # 400*10000
        logits = tf.matmul(output, softmax_w) + softmax_b

        # softmax, cross entropy 계산 처리
        # targets = [config.batch_size, config.num_steps]
        ###? weight의 의미는 무엇일까... cell 마다의 weight 값?
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits],
            targets=[tf.reshape(self.targets, [-1])],
            weights=[tf.ones([config.batch_size * config.num_steps])])
        self.cost = tf.reduce_sum(loss) / config.batch_size
        self.final_state = state

        if not is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)

        # 훈련이 가능하다고 설정한 모든 변수들
        tvars = tf.trainable_variables()

        # 여러 값들에 대한 기울기 클리핑
        # contrib.keras.backend.gradients
        # gradients gradients of variables
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        ###학습속도 설정
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


def run_epoch(session, m, data, is_training=False):
    """Run the model"""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0

    # 손실 최소 연산
    eval_op = m.train_op if is_training else tf.no_op()

    # initial_state 2*20*200
    state_list = []
    for c, h in m.initial_state:
        state_list.extend([c.eval(), h.eval()])

    #학습 데이터를 배치 개수로 나누어 num_step 만큼 읽어온다.
    ptb_iter = reader.ptb_iterator(data, m.batch_size, m.num_steps)
    for step, (x, y) in enumerate(ptb_iter):
        fetch_list = [m.cost]
        # final_state 에 담긴 상태를 꺼내 fetch_list로
        for c, h in m.final_state:
            fetch_list.extend([c, h])
        fetch_list.append(eval_op)

        # 이전 스텝에서 구한 state_list가 feed_dict로 주입
        feed_dict = {m.input_data: x, m.targets: y}
        for i in range(len(m.initial_state)):
            c, h = m.initial_state[i]
            feed_dict[c], feed_dict[h] = state_list[i * 2:(i + 1) * 2]

        # fetch_list에 담긴 final_state의 결과를 state_list로
        cost, *state_list, _ = session.run(fetch_list, feed_dict)

        costs += cost
        iters += m.num_steps

        if is_training and step % (epoch_size // 10) == 10:
            print("%.3f perplecity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)


raw_data = reader.ptb_raw_data('../simple-examples/data')
train_data, valid_data, test_data, _ = raw_data

with tf.Session() as session:

    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    # 학습, 검증, 테스트 모델
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel(config, is_training=True)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel(config)
        mtest = PTBModel(eval_config)

    tf.initialize_all_variables().run()

    # 반복 학습을 저장한다.
    saver = tf.train.Saver()
    save_dir = os.path.dirname(__file__)
    #체크포인트 파일이 있으면 변수 값을 복구한다.
    initial_step = 0
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
        print(initial_step)
    else:
        print("There is no checkpoint")

    for i in range(initial_step, config.max_max_epoch):

        # lr_decay는 반복 속도를 조절
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

        perplexity = run_epoch(session, m, train_data, is_training=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, perplexity))

        perplexity = run_epoch(session, mvalid, valid_data)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, perplexity))

        # 3 epoch마다 저장
        if i % 3 == 0:
            saver.save(session, save_dir+'LSTM.ckpt', global_step=i)

    perplexity = run_epoch(session, mtest, test_data)
    print("Test Perplexity: %.3f" % perplexity)