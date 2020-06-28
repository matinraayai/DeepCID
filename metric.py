import tensorflow as tf
import tqdm


def accuracy(model: tf.keras.Model, dataset: tf.data.Dataset):
    num_pred = 0
    correct_pred = 0
    for xs, ys in dataset:
        prediction = tf.argmax(model(xs, training=False), axis=1) == tf.argmax(ys, axis=1)
        num_pred += prediction.shape[0]
        correct_pred += tf.reduce_sum(tf.cast(prediction, tf.int32)).numpy()
    return correct_pred / num_pred