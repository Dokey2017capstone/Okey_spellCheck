#초기 가중치에 따라서 결과가 너무 크게 바뀜 정확도(1~90)
#->가중치 자동 설정 함수 사용해야 함
#->입력 단어의 가변성을 위해 padding 작업 필요
#단어 임베딩 방법
import tensorflow as tf
import numpy as np


#input character
char_arr = ['a','b','c','d','e','d','f','g','h','i','j','k','l','m','n','o','p','w','y','z']
char_dic = {n:i for i, n in enumerate(char_arr)}
input_char_num = len(char_dic)

#input word
word_arr = ['hello', 'apple', 'olleh']
word_dic = {n:i for i, n in enumerate(word_arr)}
input_word_num = len(word_dic)

#오타 교정을 위한 학습 데이터
#글자 단위 학습 -> 결과는 단어 단위
x_data = ['hella', 'hallo', 'helll','hwllo', 'appll', 'aeple','epple','olllh','oleeh','ollee' ]
y_data = ['hello','hello','hello','hello','apple','apple','apple','olleh','olleh','olleh']

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

batch_size = len(x_data)


#입력과 출력 데이터들을 one-hot encoding 해줍니다.
#tensorflow 함수가 있었던 것 같음.....확인 해보자
def one_hot(x, y):
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        #word_x = hella 오타 단어
        #word_y = hello 정답 단어
        word_x = x[i]
        word_y = y[i]

        #{h : 8} {e : 5} ... - > [7, 4, 11, 11, 0]
        x_data = [char_dic[char] for char in word_x]
        y_data = word_dic[word_y]

        #x_data = [[0 0 0 0 0 0 0 1 0 0 0...][0 0 0 0 1 0 0 0 0...]...]
        x_batch.append(np.eye(input_char_num)[x_data])
        y_batch.append(np.eye(input_word_num)[y_data])

    return x_batch, y_batch


#신경망 모델
X = tf.placeholder(tf.float32, [None, input_cell, input_size])
Y = tf.placeholder(tf.int32, [None, output_size])

W = tf.Variable(tf.random_normal([output_size, output_size]))
b = tf.Variable(tf.random_normal([output_size]))

#RNN cell
cell = tf.contrib.rnn.BasicLSTMCell(units_num, state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)
#입력 문장 길이 조절 가능
outputs, states = tf.nn.dynamic_rnn(cell = cell,inputs = X , initial_state = initial_state, dtype = tf.float32)


#손실 함수
last_output = outputs[:,-1]
logits = tf.matmul(last_output,W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)
prediction = tf.argmax(last_output, axis = 1)

#########
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch = one_hot(x_data, y_data)
for epoch in range(1000):
    _, loss = sess.run([train, cost], feed_dict={X : x_batch, Y : y_batch})
    print (sess.run(prediction , feed_dict={X:x_batch, Y : y_batch}))

#test
test = tf.argmax(last_output, 1)
test_check = tf.equal(test, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(test, tf.float32))

test_data = ['hella', 'hallo', 'helll','hwllo', 'appll', 'aeple','epple','ollwh','oleeh','ollee' ]
test_logits = ['hello','hello','hello','hello','apple','apple','apple','olleh','olleh','olleh']

x_batch, y_batch = one_hot(test_data, test_logits)

real, predict, accuracy_val = sess.run([tf.argmax(Y,1), test, accuracy],
                                       feed_dict = {X : x_batch, Y:y_batch})



print('실제값',[real])
print('예측값',[test])
print('정확도' , accuracy_val)


