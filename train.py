import argparse
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json
import datetime
from tools.metadata_tools import *


def main():
    ####### Parameter parsing #######
    parser = argparse.ArgumentParser(description='Arguments needed to prepare the metadata files')
    parser.add_argument('--trn', dest='train', help='File path to load test json files.', required=True)
    parser.add_argument('--tst', dest='test', help='File path to load test json files.', required=True)

    args = parser.parse_args()
    #################################

    train = args.train
    test = args.test
    print( train )
    print( test )

    matrix_metadata = metadata_to_matrix(train, "json")
    train_data = matrix_metadata[:,:14].astype(np.float32)
    from keras.utils import to_categorical
    train_label = to_categorical(matrix_metadata[:,14].astype(np.float32).astype(np.int8))

    matrix_metadata = metadata_to_matrix(test, "json")
    test_data = matrix_metadata[:,:14].astype(np.float32)
    from keras.utils import to_categorical
    test_label = to_categorical(matrix_metadata[:,14].astype(np.float32).astype(np.int8))

    ############## Multi Label Perceptron ##############
    # neural network hyperparameters
    num_classes = train_label.shape[1]      # hot-one encode - one column for each class
    num_epochs = 150
    batch_size = 32


    model = keras.Sequential()
    model.add(keras.layers.Dense(units=11, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))
    #model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=num_epochs, verbose=0)
    # evaluate model and show metrics
    loss, accuracy = model.evaluate(test_data, test_label, verbose=1)
    print('Test loss: ', loss, '\tTest accuracy: ', accuracy)

    model_path = 'models/trainingJSON'
    timing = datetime.datetime.now().strftime('_%Y-%m-%d_%H:%M:%S')
    # serialize model to json and save
    model_json = model.to_json()
    with open(model_path + timing + '.json', 'w') as json:
        json.write(model_json)
    # serialize weights to HDF5 and save
    model.save_weights(model_path + timing + '.h5')
    print('Model saved in:', model_path+timing)


######**********#######  MAIN  #########**********########
if __name__ == '__main__':
    main()