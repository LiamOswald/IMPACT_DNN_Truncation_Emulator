import TruncationEmulator
import Models
import keras

def main():
    #model = Models.load_model_resNet50(input_shape=(32, 32, 3))
    model = Models.load_model_alexnet(input_shape=(32, 32, 3))
    x_train, y_train, x_test, y_test = Models.load_data()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
                  metrics=['accuracy'])

    model.load_weights('AlexNet.h5')

    variables = [0, 16, 31]
    for i1 in variables:
        args = [i1, i1, i1, i1, i1, i1, i1, 0] #spesific to AlexNet, makes it so the final output layer is never truncated

        model = TruncationEmulator.truncate_weights(model, args)

        TruncationEmulator.test_model(model, x_train, y_train, x_test, y_test, args, 'AlexNet')



if __name__ == "__main__":
    main()