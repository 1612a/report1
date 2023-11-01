南華大學人工智慧期中報告
主題:初學者的 TensorFlow 2.0 教程
組員:11122111李佳楡 11023037陳蔓萱 11028001張雅琪
作業流程如下

1.首先將TensorFlow 導入程序
import tensorflow as tf

2.加載MNIST數據集。將樣本數據從整數轉換為浮點數：
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step

3.通過堆疊層来構建 tf.keras.Sequential 模型:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

4.對於每個樣本，模型都會返回一个包含 logits 或 log-odds，分數的向量，每個類一個:
predictions = model(x_train[:1]).numpy()

tf.nn.softmax 函數將這些 logits 轉換為每個類的概率：
tf.nn.softmax(predictions).numpy()
array([[0.08891758, 0.13581504, 0.04846337, 0.06362136, 0.04070919,
        0.11490263, 0.17078467, 0.07465092, 0.18853545, 0.0735998 ]],
      dtype=float32)

 5.使用 losses.SparseCategoricalCrossentropy為訓練定義損失函數，它會接受 logits 向量和 True 索引，並為每个樣本返回一个標量損失:
 loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

 6.此損失等於 true 類的負對數概率：如果模型确定類正确，。這個未經訓練的模型给出的概率接近隨機（每個類為 1/10），因此初始損失應該接近 -tf.math.log(1/10) ~= 2.3:
loss_fn(y_train[:1], predictions).numpy()
0.00093606993

7.在開始訓練之前，使用 Keras Model.compile 配置和編譯模型。 將 optimizer 類別設為 adam，將 loss 設定為您先前定義的 loss_fn 函數，並透過將 metrics 參數設為 accuracy 來指定要為模型評估的指標。
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

8.使用 Model.fit 方法調整您的模型參數並最小化損失：
model.fit(x_train, y_train, epochs=5)
Epoch 1/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0675 - accuracy: 0.9785
Epoch 2/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0586 - accuracy: 0.9811
Epoch 3/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0519 - accuracy: 0.9827
Epoch 4/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0477 - accuracy: 0.9845
Epoch 5/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0443 - accuracy: 0.9855
<keras.src.callbacks.History at 0x7c1773f06ad0>

9.Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上檢查模型效能。
model.evaluate(x_test,  y_test, verbose=2)
313/313 - 1s - loss: 0.0746 - accuracy: 0.9761 - 674ms/epoch - 2ms/step
[0.07464662194252014, 0.9761000275611877]

10.現在，這個照片分類器的準確度已經接近 98%。
如果您想讓模型返回機率，可以封裝經過訓練的模型，並將 softmax 附加到該模型：
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[3.1357499e-07, 1.7557096e-09, 1.3425333e-06, 3.1300344e-05,
        3.1137037e-11, 1.4660566e-07, 4.0756773e-15, 9.9996185e-01,
        4.1396987e-07, 4.6358919e-06],
       [4.8699695e-07, 5.7306176e-04, 9.9924350e-01, 1.2119730e-04,
        4.4787817e-13, 1.9157314e-07, 2.3823028e-05, 6.7813488e-10,
        3.7788181e-05, 3.9807841e-13],
       [2.0344415e-07, 9.9618572e-01, 1.0719226e-04, 9.6817284e-06,
        1.4588626e-05, 1.2844469e-05, 1.1701437e-05, 2.8729190e-03,
        7.8392879e-04, 1.2016635e-06],
       [9.9992836e-01, 2.1502304e-10, 4.8260958e-05, 7.0681716e-07,
        5.0750573e-06, 2.8021424e-07, 5.6977478e-06, 7.3169040e-06,
        1.6803372e-07, 4.0014793e-06],
       [3.1008385e-06, 6.3397568e-12, 7.0007246e-07, 7.8687123e-10,
        9.9846661e-01, 1.2702397e-07, 3.1732424e-07, 7.3298710e-05,
        4.8488968e-08, 1.4557794e-03]], dtype=float32)>
參考資料:
https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh_cn

