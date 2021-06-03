import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

IMG_SIZE = (224, 224)

train_path = '/home/tim/Datasets/csc532/datasets/datasets/train'
test_path = '/home/tim/Datasets/csc532/datasets/datasets/test'

train_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, zoom_range=0.2, horizontal_flip=True)
valid_gen = ImageDataGenerator(rescale=1. / 255)

train_data = train_gen.flow_from_directory(train_path,
                                           target_size=IMG_SIZE,
                                           class_mode='binary',
                                           batch_size=64, )

valid_data = valid_gen.flow_from_directory(train_path,
                                           target_size=IMG_SIZE,
                                           class_mode='binary',
                                           batch_size=64, )
# print(train_data.class_indices)

base_path = '/home/tim/trained/imagenet_mobilenet_v2_100_224_feature_vector_4'
base_model = hub.KerasLayer(base_path, input_shape=IMG_SIZE + (3,))
base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
# model.summary()

log_callback = tf.keras.callbacks.TensorBoard('/home/tim/trained/covid-19/modellog')
ckpt_dir = '/home/tim/trained/covid-19/'
ckpt_prefix = os.path.join(ckpt_dir, 'model_{epoch}.h5')
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_prefix, save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(train_data, validation_data=valid_data, epochs=10, callbacks=[log_callback, ckpt_callback, early_stopping_callback])
model.save('/home/tim/trained/covid-19/covid-19model')
