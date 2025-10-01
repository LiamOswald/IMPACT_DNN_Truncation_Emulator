from tensorflow import keras as K
import tensorflow as tf

def load_model_resNet50(input_shape=(224, 224, 3)):
    def identity_block(X, f, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters

        X_shortcut = X

        X = K.layers.Conv2D(filters=F2, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = K.layers.Activation('relu')(X)

        X = K.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = K.layers.Activation('relu')(X)

        X = K.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X = K.layers.Add()([X, X_shortcut])  # SKIP Connection
        X = K.layers.Activation('relu')(X)

        return X

    def convolutional_block(X, f, filters, stage, block, s=2):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        X_shortcut = X

        X = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = K.layers.Activation('relu')(X)

        X = K.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = K.layers.Activation('relu')(X)

        X = K.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        X_shortcut = K.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                            kernel_initializer=K.initializers.glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = K.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        X = K.layers.Add()([X, X_shortcut])
        X = K.layers.Activation('relu')(X)

        return X

    X_input = K.layers.Input(input_shape)

    #X = tf.keras.layers.UpSampling3D(size=(244,244,3))(X_input)
    X = tf.keras.layers.UpSampling2D(size=(7, 7))(X_input)

    #X = K.layers.ZeroPadding2D((3, 3))(X_input)
    X = K.layers.ZeroPadding2D((3, 3))(X)

    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=K.initializers.glorot_uniform(seed=0))(X)
    X = K.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = K.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    X = K.layers.Flatten()(X)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Dense(10, activation='softmax')(X)

    model = K.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

def load_model_alexnet(input_shape):
    model = K.models.Sequential([
        K.layers.Input(input_shape),
        K.layers.UpSampling2D(size=(7,7)),
        K.layers.ZeroPadding2D((3, 3)),
        K.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu'),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        K.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        K.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        K.layers.BatchNormalization(),
        K.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        K.layers.BatchNormalization(),
        K.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        K.layers.Flatten(),
        K.layers.Dense(4096, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(4096, activation='relu'),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax')
    ])

    return model


def load_data():
    def preprocess_data(X, Y):
        """
        a function that trains a convolutional neural network to classify the
        CIFAR 10 dataset
        :param X: X is a numpy.ndarray of shape (m, 32, 32, 3) containing the
        CIFAR 10 data, where m is the number of data points
        :param Y: Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
        labels for X
        :return: X_p, Y_p
            X_p is a numpy.ndarray containing the preprocessed X
            Y_p is a numpy.ndarray containing the preprocessed Y
        """
        X_p = K.applications.resnet50.preprocess_input(X)
        Y_p = K.utils.to_categorical(Y, 10)
        return X_p, Y_p

        return x_train, y_train, x_test, y_test

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    print((x_train.shape, y_train.shape))
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    print((x_train.shape, y_train.shape))

    return x_train, y_train, x_test, y_test