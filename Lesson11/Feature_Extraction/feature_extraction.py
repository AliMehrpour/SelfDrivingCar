import pickle
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Activation, Input
from keras.models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, 'The number of epochs')
flags.DEFINE_integer('batch_size', 128, 'The batch size')

def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    """
    How to run:
    >>> python feature_extraction.py --training_file vgg-100/vgg_traffic_100_bottleneck_features_train.p 
                                     --validation_file vgg-100/vgg_traffic_bottleneck_features_validation.p 
                                     --epochs 10 
                                     --batch_size 128
    """

    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))

    input_shape = X_train.shape[1:]
    print ('input shape -> ', input_shape)
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    
    model = Model(inp, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
