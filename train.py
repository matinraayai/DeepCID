# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 20:03:39 2018

@author: admin
"""
import tensorflow as tf
from model import DeepCID
from metric import accuracy
import matplotlib.pyplot as plt
from dataset import raman_dataset
import tqdm


def train(model: tf.keras.Model, compound=0, data_path=u'./augmented_data/',
          batch_size=300, epochs=300,
          model_path=u'./model/'):
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    train_dataset, valid_dataset, test_dataset = raman_dataset(compound,
                                                               data_path,
                                                               batch_size)

    accuracy_valid = []
    loss_valid = []
    save_file = model_path + 'component_' + str(compound) + '/component.cpt'

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    iterator = tqdm.tqdm(range(epochs))
    acc_val = 0
    loss_val = 0
    iterator.set_description("Training Progress")
    for epoch in iterator:
        for batch_xs, batch_ys in train_dataset:
            loss_val = train_step(batch_xs, batch_ys)
            iterator.set_postfix({"Loss": "%.3f" % loss_val.numpy(),
                                  "Accuracy": "%.2f" % acc_val})
        acc_val = accuracy(model, valid_dataset)

        accuracy_valid.append(acc_val)
        loss_valid.append(loss_val)

    TIMES = [(i + 1) for i in range(0, epochs)]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(TIMES, accuracy_valid, 'r')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax2 = ax1.twinx()
    ax2.plot(TIMES, loss_valid, 'g')
    ax2.set_ylabel('Loss')

    tf.keras.models.save_model(model, save_file)
    print("Saved model at %s." % save_file)


if __name__ == '__main__':
    m = DeepCID(0.2)
    train(m)
