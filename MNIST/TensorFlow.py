from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

path = os.path.dirname( os.path.realpath(__file__) )
dir_path = "./saves"

def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,export_outputs=export_outputs)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    traingDataCSV = csv.reader( open(path + "/mnist_train.csv", "r") )
    traingData = []

    for row in traingDataCSV:
            traingData.append( [int(i) for i in row] )

    testingDataCSV = csv.reader( open(path + "/mnist_test.csv", "r") )
    testingData = []

    for row in testingDataCSV:
            testingData.append( [int(i) for i in row] )


    train_data = np.asarray(traingData, dtype=np.float32)[:, 1:]
    train_labels = np.asarray(traingData, dtype=np.int32)[:, 0]

    eval_data = np.asarray(testingData, dtype=np.float32)[:, 1:]
    eval_labels = np.asarray(testingData, dtype=np.int32)[:, 0]

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=dir_path)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    x = tf.feature_column.numeric_column("x")
    feature_columns = [x]
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    mnist_classifier.export_savedmodel(dir_path, export_input_fn)

tf.app.run()
