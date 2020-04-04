# 텐서플로우 2.0 버전 이전의 버전을 사용하기 위해 사용
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from tensorflow import keras
from datetime import datetime

# 그래프 리셋
tf.reset_default_graph()

# 랜덤값 생성
np.random.seed(20191209)
tf.set_random_seed(20191209)

# load data / keras 데이터 로드
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

x = tf.placeholder(tf.float32, [None, 784])  # 입력되는 이미지 데이터

w2 = tf.Variable(tf.truncated_normal([784, 300]))  # num_units1은 입력값 (64*7*7), num_unit2는 뉴런의 수
b2 = tf.Variable(tf.constant(0.1, shape=[300]))

# 1024개의 은닉층, ReLu 적용
hidden2 = tf.nn.relu(tf.matmul(x, w2) + b2)

# 소프트맥스 함수를 이용하여 10개의 카테고리로 분류
w0 = tf.Variable(tf.zeros([300, 10]))
b0 = tf.Variable(tf.zeros([10]))
k = tf.matmul(hidden2, w0) + b0  # k는 소프트맥스 층을 적용하기 전의 값
p = tf.nn.softmax(k)

# define loss (cost) function
# 비용 함수 정의
t = tf.placeholder(tf.float32, [None, 10])  # 플레이스 홀더로 정의. 나중에 학습 데이터 셋에서 읽을 라벨
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = k, labels = t))  # tf.nn.softmax_cross_entropy_with_logits 함수는 softmax가 포함되어 있는 함수
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)  # 비용 함수를 최적화 하기 위해서 최적화 함수 AdamOptimizer 사용

# 정확도 계산 함수
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))  # 학습 결과와 입력된 라벨(정답)을 비교하여 맞았는지 틀렸는지를 리턴
# argmax는 인자에서 가장 큰 값의 인덱스를 리턴함. 0~9 배열이 들어가 있기 때문에 가장 큰 값이 학습에 의해 예측된 숫자
# p는 예측의 결과값, t는 학습의 결과(라벨)값. 두 값을 비교하여 가장 큰 값이 있는 인덱스가 일치하면 예측이 성공한 것
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# correct_predication이 bool 값이기 때문에 이 값을 숫자로 바꾸고 저장
tf.summary.scalar('accuracy', accuracy) # 정확도 모니터링을 위해 accuracy 사용

# 텐서보드
summary_init = tf.summary.merge_all() # summary 사용을 위한 초기화

# prepare session
# 학습 세션을 시작하고 변수를 초기화
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# 텐서보드
summary_writer = tf.summary.FileWriter("mlp_tensorboard_logs", sess.graph) # 파일 경로를 선택하고, 세션 그래프를 인자로 넘김

# start training
# 배치 학습 시작
i = 0
startTime = datetime.now() # 학습 시간 측정
for _ in range(1000):
    i += 1
    # batch_xs에는 학습에 사용할 28*28*1 사이즈의 이미지
    # batch_ts에는 그 이미지에 대한 라벨
    batch_xs = train_images[(i - 1) * 50: i * 50].reshape(-1, 28 * 28) / 255.0  # 일자로 피고 255로 나눔
    batch_ts = tf.one_hot(train_labels[(i - 1) * 50: i * 50], depth=10).eval(session=tf.Session())
    summary, _ = sess.run([summary_init, train_step], feed_dict={x: batch_xs, t: batch_ts})  # feed_dict를 통해 피딩(입력)하고 트레이닝 세션을 시작함
    # 마지막 인자에 keep_prob를 0.5로 정하여 50%의 네트워크를 인위적으로 끊음
    summary_writer.add_summary(summary, global_step = i)
    
    # 100번마다 중간중간 정확도와 학습 비용을 계산하여 출력함
    if i % 100 == 0:
        loss_vals, acc_vals = [], []
        # 한번에 검증하지 않고 테스트 데이터를 4등분 한 후 4분의 1씩 테스트 데이터를 로딩해서 학습 비용과 학습 정확도를 계산함
        for c in range(4):
            start = len(test_labels) / 4 * c
            end = len(test_labels) / 4 * (c + 1)
            # numpy 현재 버전에서는 부동소수점을 지원하지 않기 때문에 int형으로 변환
            start = int(start)
            end = int(end)
            loss_val, acc_val = sess.run([loss, accuracy],
                                         feed_dict={x: test_images[start:end].reshape(-1, 28 * 28) / 255.0,
                                                    t: tf.one_hot(test_labels[start:end], depth=10).eval(session=tf.Session())})

            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        # 여기까지 데이터 검증
        loss_val = np.sum(loss_vals)
        acc_val = np.mean(acc_vals)
        print('Step: %d, Loss: %f, Accuracy: %f'
              % (i, loss_val, acc_val))

print('AllTime : ', (datetime.now()) - startTime)   # 걸린 시간 출력

# 학습 결과 저장
saver.save(sess, 'cnn_session')
sess.close()