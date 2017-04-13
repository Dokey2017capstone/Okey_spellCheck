#초기 가중치에 따라서 결과가 너무 크게 바뀜 정확도(1~90)
#->가중치 자동 설정 함수 사용해야 함
#->입력 단어의 가변성을 위해 padding 작업 필요
#단어 임베딩 방법
import tensorflow as tf
import numpy as np
import random

#input character
char_arr = ['a','b','c','d','e','d','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
char_dic = {n:i for i, n in enumerate(char_arr)}
input_char_num = len(char_arr)

#input word
word_arr = ['hello', 'apple', 'olleh','mommy','happy','money','schoo','puppy','sunny','black']
word_dic = {n:i for i, n in enumerate(word_arr)}
input_word_num = len(word_arr)

#오타 교정을 위한 학습 데이터
#글자 단위 학습 -> 결과는 단어 단위
x_data = ['yello','hella', 'hallo', 'helll','hwllo',
          'appll', 'aeple','epple','qpple','zpple',
          'olley','olllh','oleeh','ollee','olpeh',
          'momny','mommu','mpmmy','nommy','mimmy',
          'happy','halpy','hqppy','haopy','hapoy',
          'mobey','mpney','noney','monei','monwy',
          'schio','scgoo','achoo','sxhoo','schpo',
          'pupoy','luppy','pyppy','puopy','pupoy',
          'sinny','synny','eunny','sunby','sunmy',
          'bcack','bkack','blwck','blavk','blacl',
          'yello', 'hella', 'hallo', 'helll', 'hwllo',
          'appll', 'aeple', 'epple', 'qpple', 'zpple',
          'olley', 'olllh', 'oleeh', 'ollee', 'olpeh',
          'momny', 'mommu', 'mpmmy', 'nommy', 'mimmy',
          'happy', 'halpy', 'hqppy', 'haopy', 'hapoy',
          'mobey', 'mpney', 'noney', 'monei', 'monwy',
          'schio', 'scgoo', 'achoo', 'sxhoo', 'schpo',
          'pupoy', 'luppy', 'pyppy', 'puopy', 'pupoy',
          'sinny', 'synny', 'eunny', 'sunby', 'sunmy',
          'bcack', 'bkack', 'blwck', 'blavk', 'blacl'
          ]
y_data = ['hello', 'hello', 'hello', 'hello', 'hello',
          'apple','apple','apple','apple','apple',
          'olleh','olleh','olleh','olleh','olleh',
          'mommy','mommy','mommy','mommy','mommy',
          'happy','happy','happy','happy','happy',
          'money','money','money','money','money',
          'schoo','schoo','schoo','schoo','schoo',
          'puppy','puppy','puppy','puppy','puppy',
          'sunny','sunny','sunny','sunny','sunny',
          'black','black','black','black','black',
          'hello', 'hello', 'hello', 'hello', 'hello',
          'apple', 'apple', 'apple', 'apple', 'apple',
          'olleh', 'olleh', 'olleh', 'olleh', 'olleh',
          'mommy', 'mommy', 'mommy', 'mommy', 'mommy',
          'happy', 'happy', 'happy', 'happy', 'happy',
          'money', 'money', 'money', 'money', 'money',
          'schoo', 'schoo', 'schoo', 'schoo', 'schoo',
          'puppy', 'puppy', 'puppy', 'puppy', 'puppy',
          'sunny', 'sunny', 'sunny', 'sunny', 'sunny',
          'black', 'black', 'black', 'black', 'black'
          ]

#####################
#rnn 셀 하나에 들어갈 one-hot 원소 수
input_size = input_char_num
#rnn 입력 전체 셀
input_cell = 5

#분류될 클래스 수
units_num = input_word_num
output_size = input_word_num
#many to one
output_cell = 1

batch_size = 20


#입력과 출력 데이터들을 one-hot encoding 해줍니다.
#tensorflow 함수가 있었던 것 같음.....확인 해보자
def one_hot(x, y):
    x_batch = []
    y_batch = []
    num = random.randrange(0,20)
    x_ = x[num*5: num*5 + 5]
    y_ = y[num*5: num*5 + 5]

    for i in range(5):
        #word_x = hella 오타 단어
        #word_y = hello 정답 단어

        word_x = x_[i]
        word_y = y_[i]

        #{h : 8} {e : 5} ... - > [7, 4, 11, 11, 0]
        x_d = [char_dic[char] for char in word_x]
        y_d = word_dic[word_y]


        #x_data = [[0 0 0 0 0 0 0 1 0 0 0...][0 0 0 0 1 0 0 0 0...]...]
        x_batch.append(np.eye(input_char_num)[x_d])
        y_batch.append(np.eye(input_word_num)[y_d])

    return x_batch, y_batch


#신경망 모델
X = tf.placeholder(tf.float32, [None, 5,input_char_num])
Y = tf.placeholder(tf.int32, [None, output_size])

W = tf.Variable(tf.random_normal([input_char_num, output_size]))
b = tf.Variable(tf.random_normal([output_size]))

#RNN cell
num_hidden = 128
cell = tf.contrib.rnn.BasicLSTMCell(units_num, state_is_tuple = True)
cell = tf.contrib.rnn.MultiRNNCell([cell]*2, state_is_tuple = True)
#입력 문장 길이 조절 가능
hiddens = tf.contrib.layers.fully_connected(X, num_hidden, activation_fn = tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell = cell,inputs = hiddens ,  dtype = tf.float32)
outputs = tf.contrib.layers.fully_connected(outputs, input_char_num, activation_fn = None)

#손실 함수
last_output = outputs[:,-1]
logits = tf.matmul(last_output,W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
prediction = tf.argmax(logits, axis = 1)

#########
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(2000):
    x_batch, y_batch = one_hot(x_data, y_data)
    _, loss = sess.run([train, cost], feed_dict={X : x_batch, Y : y_batch})

#test
test = prediction
test_check = tf.equal(test, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(test_check, tf.float32))

test_data = ['hrllo', 'spple','nommy','puopy','ouleh',
          'mobey','svhoo', 'ouppy', 'sunjy', 'gkack',
          'heilo', 'aople','mlmmy','pupoy','plleh',
          'mobey','scnoo', 'pippy', 'suuny', 'glack',
          'heklo', 'aople','mpmmy','pkppy','olkeh',
          'mobey','scyoo', 'puopy', 'wunny', 'blavk',
          'fello', 'alple','mojmy','pyppy','olley',
          'noney','scjoo', 'pulpy', 'dunny', 'bkack',
          'gello', 'apole','mokmy','pulpy','ollah',
          'monwy','schpo', 'puopy', 'sknny', 'bkack',
          'helko', 'apkle','momny','puopy','ollrh',
          'monry','schuo', 'pupoy', 'synny', 'bkack',
          'heklo', 'spple','nommy','pulpy','oleeh',
          'momey','schpo', 'phppy', 'shnny', 'vlack',
          'hellp', 'apole','monmy','pupoy','plleh',
          'moneu','schoi', 'puppt', 'sunnt', 'blacl',
          'helko', 'appke','momny','pupoy','ollrh',
          'momey','scjoo', 'puopy', 'subny', 'blsck',
          'hrllo', 'aople','mimmy','pippy','ooleh',
          'noney','echoo', 'ouppy', 'dunny', 'vlack']
test_logits = ['hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black',
          'hello', 'apple','mommy','puppy','olleh',
          'money','schoo', 'puppy', 'sunny', 'black'     ]

result_accur = 0

for i in range(50):
    x_batch, y_batch = one_hot(test_data, test_logits)

    real, predict, c = sess.run([tf.argmax(Y,1), test, accuracy],
                                           feed_dict = {X : x_batch, Y:y_batch})


    print('목표값',[test_logits[i] for i in real])
    print('오타값',[test_data[i] for i in predict])
    print('예측값',[test_logits[i] for i in predict])
    result_accur += c
print('정확도', result_accur/50)

