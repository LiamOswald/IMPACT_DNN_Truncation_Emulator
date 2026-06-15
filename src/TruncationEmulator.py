import numpy as np
import os
from codecs import decode
import struct


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Disables GPU acceleration

global count_total_bits_truncated
count_total_bits_truncated = 0

def read_args():
    f = open('args.txt')
    args = f.read().split('\n')
    f.close()
    return args


def truncate_weights(model, args):
    def bin_truncation(binary_string, n_bits):
        if n_bits == 0: #Error Condition avoidance
            return binary_string

        #fill truncation values
        result = binary_string[:-n_bits] + '1'
        while (len(result) < 32):
            result = result + '0'

        return (result)

    def bin_to_float(b):
        """ Convert binary string to a float. """
        b = b + '00000000000000000000000000000000'  # convert back to 64 bit
        bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
        return struct.unpack('>d', bf)[0]

    def int_to_bytes(n, length):  # Helper function
        """ Int/long to byte string.

            Python 3.2+ has a built-in int.to_bytes() method that could be used
            instead, but the following works in earlier versions including 2.x.
        """
        return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]

    def float_to_bin(value):  # For testing.
        """ Convert float to 32-bit binary string. """
        [d] = struct.unpack(">Q", struct.pack(">d", value))
        return '{:064b}'.format(d)[:32]

    def truncation_process(weight, n_bits):
        a = float_to_bin(weight)
        b = bin_truncation(a, n_bits)
        return bin_to_float(b)

    def recursive_truncate(node, n_bits):
        #if list, dig deeper in recursion
        if 'list' in str(type(node)):
            for count, i in enumerate(node):
                node[count] = recursive_truncate(node[count], n_bits)
            return node
        #if ndarray, dig deeper in recursion
        elif 'ndarray' in str(type(node)):
            for count, i in enumerate(node):
                node[count] = recursive_truncate(node[count], n_bits)
            return node
        #finally, if single weight. truncate and return up recusion
        else:
            global count_total_bits_truncated
            count_total_bits_truncated = count_total_bits_truncated + n_bits
            return truncation_process(node, n_bits)

    def truncation_handler(layer, n_bits):
        def convolution_handler(layer):
            weights = layer.get_weights()
            print('CNN truncation: n_bits = ', n_bits)
            new_weights = recursive_truncate(weights, n_bits)
            return new_weights

        def dense_handler(layer):
            weights = layer.get_weights()
            print('ANN truncation: n_bits = ', n_bits)
            new_weights = recursive_truncate(weights, n_bits)
            return new_weights


        s = str(type(layer))

        if 'Conv2D' in s:
            print('Convolution')
            return convolution_handler(layer)

        elif 'Dense' in s:
            print('DENSE')
            return dense_handler(layer)

        else:
            print(type(layer))
            print(layer.get_weights())
            print(s)
            print('LAYER UNEXPECTED')
            exit()

    def check_skip_layers(layer):
        s = str(type(layer))
        if 'convolutional.UpSampling2D' in s:
            print('action: PASS - up sampling')
            return True

        elif 'ZeroPadding2D' in s:
            print('action: PASS - ZeroPadding2D')
            return True

        elif 'MaxPooling2D' in s:
            print('actoin: PASS - MaxPooling2D')
            return True

        elif 'Dropout' in s:
            print('action: PASS - dropout')
            return True

        elif 'Flatten' in s:
            print('action: PASS - Flatten')
            return True
        elif 'UpSampling' in s:
            print('action: PASS - UpSampiling')
            return True
        elif 'BatchNormalization' in s:
            print('BATCH NORMALIZATION')
            return True

        else:
            return False


    #begin truncation
    layerTruncationIndex = 0

    print('Model has Layers = ', len(model.layers))
    for count, layer in enumerate(model.layers):
        if check_skip_layers(layer):
            pass
        else:
            #get n_bits via a global counter
            n_bits = int(args[layerTruncationIndex])
            new_layer = truncation_handler(layer, n_bits)
            model.layers[count].set_weights(new_layer)
            layerTruncationIndex = layerTruncationIndex + 1

        print('*************************************')


    return model


def test_model(model, x_train, y_train, x_test, y_test, args, file_name='Model'):

    print('Begin Testing....')
    total_correct = 0
    total_possible = y_train.shape[0] + y_test.shape[0]

    predictions = np.argmax(model.predict(x_train), axis=-1)
    for i in range(predictions.shape[0]):
        if (predictions[i] == np.argmax(y_train[i])):
            total_correct = total_correct + 1

    predictions = np.argmax(model.predict(x_test), axis=-1)

    for i in range(predictions.shape[0]):
        if (predictions[i] == np.argmax(y_test[i])):
            total_correct = total_correct + 1

    print('Total Correct: ', total_correct)
    print('Total Possibl: ', total_possible)

    print('Accuracy = ', (total_correct/total_possible))
    print(count_total_bits_truncated)

    #Save all of the important information into csv file
    for i in args:
        file_name = file_name + '_' + str(i)
    file_name = file_name + '.csv'

    f = open(file_name, "a")
    for i in args:
        f.write(str(i) + ',')
    f.write(str(total_correct) + ',' + str(total_possible) + ',' + str(count_total_bits_truncated))
    f.close()

