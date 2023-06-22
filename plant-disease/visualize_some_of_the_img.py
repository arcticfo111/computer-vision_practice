import tensorflow as tf
from tensorflow.python.keras import models, layers
import matplotlib.pyplot as plt

# 이미지 데터 tf dataset으로  불러오기, 이미지 데이터 세트를 사전처리한다. 
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

class_names = dataset.class_names # 폴더 이름('Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy') -> 클래스 이름
plt.figure(figsize=(10, 10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")
        # plt.show()