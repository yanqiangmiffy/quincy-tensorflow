#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 

"""
from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# embedding_layer相当于初始化一个1000行，5列的词嵌入矩阵
embedding_layer=layers.Embedding(1000,5)

# 下行1,2,3是获取第2行，第3行和第4行的词向量
result=embedding_layer(tf.constant([1,2,3]))
print(result.numpy())

# 输入也可以是一个序列
result = embedding_layer(tf.constant([[0,1,2],[3,4,5]]))
print(result.shape)

# Learning embeddings from scratch

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
print(encoder.subwords[:20])
train_batches = train_data.shuffle(1000).padded_batch(batch_size=10,padded_shapes=20,padding_values=0)
test_batches = test_data.shuffle(1000).padded_batch(batch_size=10,padded_shapes=20,padding_values=0)


train_batch, train_labels = next(iter(train_batches))
print(train_batch.numpy())

embedding_dim=16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1)
])

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)