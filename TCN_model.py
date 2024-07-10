import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Add, MaxPooling1D, Flatten, Concatenate, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pandas as pd


# Residual Connection
def residual_block(inputs, filters, kernel_size=5, strides=1, dilation_rate=1, dropout_rate=0.2):
    # 1D Convolution
    conv = Conv1D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='causal',
                  dilation_rate=dilation_rate,
                  activation='relu')(inputs)

    # Dropout
    dropout = Dropout(dropout_rate)(conv)

    # Normalization (Optional)
    norm = BatchNormalization()(dropout)

    # Residual connection
    residual = conv if strides == 1 and inputs.shape[-1] == filters else Conv1D(filters=filters, kernel_size=1,
                                                                                strides=strides)(inputs)

    # Add residual connection
    output = Add()([norm, residual])

    # Apply activation after residual connection (Optional)
    output = tf.keras.activations.relu(output)

    return output

# Define a model for processing time series
def build_tcn_model(input_shape, num_filters, num_blocks):
    inputs = Input(shape=input_shape)
    x = inputs

    # Stack residual blocks
    for i in range(num_blocks):
        dilation_rate = 2 ** i
        x = residual_block(x, num_filters,dilation_rate=dilation_rate)

        # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Output layer
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(2, activation='sigmoid')(x)  # use 'softmax' for classification

    model = Model(inputs=inputs, outputs=outputs)
    return model, outputs


# Extract data as required
def attain_feature_data(duration=120):
    '''
    :param duration: Determine the duration of the segmentation, the longest is 120 minutes
    :return: Feature dataset X, label dataset y
    '''
    timestamp = 120
    # Get sample feature vector X and label vector y
    X_attaind = []
    y = []
    with open('Train_Test_dataset.json', 'r') as f:
        data = json.load(f)

    for line in data:
        label = line['lable']
        if label == 'ASleak':
            y.append(0)
        elif label == 'manrs':
            y.append(1)
        else:
            continue
        nb_A_temp = line['nb_A']
        nb_dup_A_temp = line['nb_dup_A']
        nb_implicit_W_temp = line['nb_implicit_W']
        nb_A_prefix_temp = line['nb_A_prefix']
        max_A_prefix_temp = line['max_A_prefix']
        avg_A_prefix_temp = line['avg_A_prefix']
        nb_new_A_afterW_temp = line['nb_new_A_afterW']
        max_path_len_temp = line['max_path_len']
        avg_path_len_temp = line['avg_path_len']
        max_editdist_temp = line['max_editdist']
        avg_editdist_temp = line['avg_editdist']
        nb_tolonger_temp = line['nb_tolonger']
        nb_toshorter_temp = line['nb_toshorter']

        nb_A = nb_A_temp[timestamp-duration:timestamp+duration]
        nb_dup_A = nb_dup_A_temp[timestamp-duration:timestamp+duration]
        nb_implicit_W = nb_implicit_W_temp[timestamp-duration:timestamp+duration]
        nb_A_prefix = nb_A_prefix_temp[timestamp-duration:timestamp+duration]
        max_A_prefix = max_A_prefix_temp[timestamp-duration:timestamp+duration]
        avg_A_prefix = avg_A_prefix_temp[timestamp-duration:timestamp+duration]
        nb_new_A_afterW = nb_new_A_afterW_temp[timestamp-duration:timestamp+duration]
        max_path_len = max_path_len_temp[timestamp-duration:timestamp+duration]
        avg_path_len = avg_path_len_temp[timestamp-duration:timestamp+duration]
        max_editdist = max_editdist_temp[timestamp-duration:timestamp+duration]
        avg_editdist =avg_editdist_temp[timestamp-duration:timestamp+duration]
        nb_tolonger = nb_tolonger_temp[timestamp-duration:timestamp+duration]
        nb_toshorter = nb_toshorter_temp[timestamp-duration:timestamp+duration]

        # Data normalization.
        normalization_feature_temp = np.array([nb_A,nb_dup_A,nb_implicit_W,nb_A_prefix,max_A_prefix,avg_A_prefix,
                                        nb_new_A_afterW,max_path_len,avg_path_len,max_editdist,avg_editdist,
                                      nb_tolonger,nb_toshorter])
        data_min = np.min(normalization_feature_temp, axis=0, keepdims=True)
        data_max = np.max(normalization_feature_temp, axis=0, keepdims=True)
        normalization_feature = (normalization_feature_temp - data_min) / (data_max - data_min + 1e-7)

        # Traverse the transposed DataFrame row by row and convert the two-dimensional list into a NumPy array
        df = pd.DataFrame(normalization_feature)
        df_transposed = df.T
        row_lists = [list(row) for index, row in df_transposed.iterrows()]
        data_array = np.array(row_lists)
        X_attaind.append(data_array)
    return X_attaind,y


def main(duration=120,num_filters = 64,num_blocks = 4):
    # Get sample feature vector X and label vector y
    X_attaind ,y = attain_feature_data(duration=duration)

    # Divide the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_attaind, y, test_size=0.2, random_state=42)
    y_train_nn = to_categorical(y_train)
    y_test_nn = to_categorical(y_test)

    # Set the parameters used by the model
    input_shape = (duration, 13)  # Time series, duration time slices, 13 features per time slice


    # Create, compile, and train sub-models
    temporal_model, outputs = build_tcn_model(input_shape,num_filters,num_blocks)
    temporal_model.compile(optimizer='adam',
                  loss='Binary Cross-Entropy',
                  metrics=['accuracy'])
    # temporal_model.summary()
    temporal_model.fit(X_train, y_train_nn, epochs=50, batch_size=32, validation_data=(X_test, y_test_nn))

    # Evaluating the Model
    loss, accuracy = temporal_model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # y_test is the true label, y_pred is the model prediction result
    y_test_int = np.argmax(y_test, axis=1)  # If y_test is one-hot encoded
    y_pred = temporal_model.predict(X_test)
    print(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate other evaluation indicators separately
    precision = precision_score(y_test_int, y_pred_classes, average='weighted')
    recall = recall_score(y_test_int, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_int, y_pred_classes, average='weighted')

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ =='__main__':
    main()






