# 텐서플로우 2.0 버전 이전의 버전을 사용하기 위해 사용
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from tensorflow import keras
from datetime import datetime

# 그래프 리셋
tf.reset_default_graph()

# 랜덤값 생성
np.random.seed(20140114)
tf.set_random_seed(20191209)

# load data / keras 데이터 로드
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# 첫번째 컨볼루션 계층
# 필터 정의
num_filters1 = 32

x = tf.placeholder(tf.float32, [None, 784])  # 입력되는 이미지 데이터
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 784*n개인 이미지 데이터 x를 reshape로 변경한 것
# 28*28*1의 행렬을 무한개(-1)로 정의
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters1], stddev=0.1))
# 5 * 5 * 1 필터를 사용할 것이고, 필터의 수가 32개이기 때문에 차원은 [5, 5, 1, 32]가 됨

# 필터를 입력 데이터(이미지)에 적용
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
# 28 * 28 * 1 사이즈의 입력 데이터인 x_image에 필터 W_conv1을 적용함
# strides의 2번째는 가로 스트라이드 값, 3번째는 세로 스트라이드 값
# padding을 SAME으로 주면 텐서플로우가 자동으로 패딩을 삽입
# padding을 VALID로 주면 패딩을 적용하지 않고 필터를 적용하여 출력값의 크기가 작아짐

# 활성함수의 적용
# bias 값 (y = W*X+b에서 b)인 b_conv1을 정의하고, tf.nn.relu를 이용하여, 필터된 결과(h_conv1)에
# bias 값을 더한 값을 ReLu 함수로 적용함
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))  # bias 값 정의
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)  # 필터된 결과값

# max pooling 적용
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# tf.nn.max_pool 함수로 적용. 첫번째 인자는 활성화 함수 ReLu를 적용하고 나온 결과값
# ksize는 풀링 필터의 사이즈로 2*2 크기로 묶어서 풀링함

# 두번째 컨볼루션 계층 / 위와 매우 유사
# 필터 적용
num_filters2 = 64

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, num_filters1, num_filters2], stddev=0.1))
# 필터의 사이즈가 5*5이고, 입력되는 값이 32개이기 때문에 32가 들어가고 총 64개의 필터를 적용
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')

b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Max pooling 역시 첫번쨰 필터와 마찬가지로 2,2 사이즈의 필터와 2,2 stride를 적용해서
# 가로 세로로 두칸씩 움직이게 하여 결과의 크기가 반으로 줄어들게 함
# 14 * 14 크기의 입력값 32개가 들어가서 7*7 크기의 행렬 64개가 됨


# define fully connected layer
# 풀리 커넥티드 계층
# 두 개의 컨볼루션 계층을 통해서 뽑아낸 특징을 가지고 입력된 이미지가 0~9 중 어느 숫자인지
# 풀리 커넥티드 계층을 통해서 판단
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * num_filters2])  # 입력된 64개의 7 * 7 행렬을 1차원 행렬로 변환

num_units1 = 7 * 7 * num_filters2
num_units2 = 1024

# 풀리 커넥티드 레이어에 삽입. 이때 입력값은 64 * 7 * 7 개의 벡터값을 1024개의 뉴런을 이용해 학습함
w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))  # num_units1은 입력값 (64*7*7), num_unit2는 뉴런의 수
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))

# 1024개의 은닉층, ReLu 적용
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

# 드롭 아웃 적용
keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)  # tf.nn.dropout 함수로 hidden2를 넣고, keep_prob에 연결 비율을 넣으면 됨
# 연결 비율은 네트워크 전체가 다 연결되어 있으면 1, 50%를 드롭아웃 시키면 0.5

# 드롭 아웃의 결과를 가지고 소프트맥스 함수를 이용하여 10개의 카테고리로 분류
w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
k = tf.matmul(hidden2_drop, w0) + b0  # k는 소프트맥스 층을 적용하기 전의 값
p = tf.nn.softmax(k)

# define loss (cost) function
# 비용 함수 정의
t = tf.placeholder(tf.float32, [None, 10])  # 플레이스 홀더로 정의. 나중에 학습 데이터 셋에서 읽을 라벨
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = k, labels = t))  # tf.nn.softmax_cross_entropy_with_logits 함수는 softmax가 포함되어 있는 함수
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)  # 비용 함수를 최적화 하기 위해서 최적화 함수 AdamOptimizer 사용

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
summary_writer = tf.summary.FileWriter("cnn_tensorboard_logs", sess.graph) # 파일 경로를 선택하고, 세션 그래프를 인자로 넘김

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
    summary, _ = sess.run([summary_init, train_step], feed_dict={x: batch_xs, t: batch_ts, keep_prob: 0.5})  # feed_dict를 통해 피딩(입력)하고 트레이닝 세션을 시작함
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
                                                    t: tf.one_hot(test_labels[start:end], depth=10).eval(session=tf.Session()),
                                                    keep_prob: 1.0})

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
