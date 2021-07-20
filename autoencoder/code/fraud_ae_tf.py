import argparse, os
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers

from keras.utils import multi_gpu_model

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    # read in data
    train_x = pd.read_csv(os.path.join(args.training, 'train_normal.csv'))
    val_df = pd.read_csv(os.path.join(args.validation, 'validation_normal.csv'))
    val_x = val_df.loc[val_df['Class'] == 0]
    val_x = val_x.drop(['Class'], axis=1)
    val_y = val_df['Class']
    val_xall = val_df.drop(['Class'], axis=1)
    
    # build network 
    input_dim = train_x.shape[1]
    encoding_dim = int(input_dim / 2) - 1
    hidden_dim = int(encoding_dim / 2)

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation=args.activation, activity_regularizer=regularizers.l1(args.learning_rate))(input_layer)
    encoder = Dense(hidden_dim+3, activation=args.activation)(encoder)
    encoder = Dropout(args.dropout)(encoder)
    encoder = Dense(hidden_dim, activation=args.activation)(encoder)
    encoder = Dropout(args.dropout)(encoder)
    decoder = Dense(hidden_dim+3, activation=args.activation)(encoder)
    decoder = Dense(encoding_dim, activation=args.activation)(decoder)
    decoder = Dense(input_dim, activation='linear')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    print(autoencoder.summary())
    
    if args.gpu_count > 1:
        autoencoder = multi_gpu_model(autoencoder, gpus=args.gpu_count)
    
    # compile and fit 
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(train_x, train_x,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        shuffle=True,
                        validation_data=(val_x.values, val_x.values),
                        verbose=2)
        
    score = autoencoder.evaluate(val_x.values, val_x.values, verbose=0)
    print('Validation loss    :', score)
    
    # save Keras model for Tensorflow Serving
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(args.model_dir, 'model/1'),
        inputs={'inputs': autoencoder.input},
        outputs={t.name: t for t in autoencoder.outputs})
    