import tensorflow as tf
from tensorflow.python.keras import models, layers
import matplotlib.pyplot as plt

# 이미지 데터 tf dataset으로  불러오기, 이미지 데이터 세트를 사전처리
plant_img_path = './plant-data'
img_size = 256 # 이미지 크기 지정
batch_size = 32 # 표준 배치 크기
epochs = 50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    plant_img_path,
    shuffle=True, # 임의로 섞기
    image_size=(img_size, img_size),
    batch_size = batch_size 
)

## 학습 테스트 분할(80% 데이터 학습용, 10% 유효성 검사용, 10% 테스트용) 
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):
    dataset_len = len(dataset)
    train_size = int(train_split * dataset_len)
    val_size = int(val_split * dataset_len)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(img_size, img_size),
    layers.experimental.preprocessing.Rescaling(1.0 / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# cnn 구축, 훈련, 정확도 측정
