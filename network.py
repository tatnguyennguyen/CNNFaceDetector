from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization

window_size = 29
step_size = 4


def get_network_1():
    """Return network object with random weight."""
    model = Sequential()
    # 29. image
    model.add(Conv2D(16, (5, 5), padding='valid', strides=(2, 2), activation='relu', input_shape=(None, None, 3)))
    # 13
    model.add(Conv2D(16, (5, 5), padding='valid', strides=(2, 2), activation='relu'))
    # 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    # 3. feature to SVM
    model.add(Conv2D(1, (3, 3), padding='valid', strides=(1, 1), activation='linear'))  # this is y=w*x+b in SVM
    # 1
    return model


def get_network_2():
    """Return network object with random weight."""
    model = Sequential()
    # 29. image
    model.add(Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu', input_shape=(window_size, window_size, 3)))
    # 15
    model.add(Conv2D(64, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    # 11
    model.add(Conv2D(64, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    # 7
    model.add(Conv2D(96, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    # 5
    model.add(Conv2D(96, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    # 3. feature to SVM
    model.add(Conv2D(1, (3, 3), padding='valid', strides=(1, 1), activation='linear'))  # this is y=w*x+b in SVM
    # 1
    return model


def get_network_3():
    """Return network object with random weight."""
    model = Sequential()
    # 29. image
    model.add(Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu', input_shape=(window_size, window_size, 3)))
    # 15
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    # 13
    model.add(Conv2D(96, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    # 11
    model.add(Conv2D(96, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    # 9
    model.add(Conv2D(96, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    # 7
    model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    # 5
    model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    # 3
    model.add(Conv2D(256, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    # 1. feature to SVM
    model.add(Conv2D(1, (1, 1), padding='valid', strides=(1, 1), activation='linear'))  # this is y=w*x+b in SVM
    # 1
    return model


def load_network_1(weight_file):
    """Return network object with weights were loaded from weight_file."""
    model = get_network_1()
    model.load_weights(weight_file)
    return model


def load_network_2(weight_file):
    """Return network object with weights were loaded from weight_file."""
    model = get_network_2()
    model.load_weights(weight_file)
    return model


def load_network_3(weight_file):
    """Return network object with weights were loaded from weight_file."""
    model = get_network_3()
    model.load_weights(weight_file)
    return model
