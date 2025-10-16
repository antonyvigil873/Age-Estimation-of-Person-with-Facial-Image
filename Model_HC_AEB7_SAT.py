from Evaluate_error import evaluate_error
import numpy as np
from keras import Model, Input
from keras.applications import EfficientNetB7
from keras.layers import Dense, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, Concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def Model_HC_AEB7_SAT(Feature, Images, Target, sol=None):
    print('Model HC_AEB7_SAT')
    if sol is None:
        sol = [5, 5, 5]
    IMG_SIZE = 32
    Train_Datas_1D, Test_Datas_1D, Train_Targets, Test_Target = train_test_split(Feature, Target,
                                       random_state=104,
                                       test_size=0.25,
                                       shuffle=True)
    Train_Datas_2D, Test_Datas_2D, Train_Targets, Test_Target = train_test_split(Images, Target,
                                       random_state=104,
                                       test_size=0.25,
                                       shuffle=True)

    Train_x_2D = np.zeros((Train_Datas_2D.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Datas_2D.shape[0]):
        temp = np.resize(Train_Datas_2D[i], (IMG_SIZE, IMG_SIZE, 3))
        Train_x_2D[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_x_2D = np.zeros((Test_Datas_2D.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Datas_2D.shape[0]):
        temp = np.resize(Test_Datas_2D[i], (IMG_SIZE, IMG_SIZE, 3))
        Test_x_2D[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    # Prepare 1D input data (features)
    Train_x_1D = np.asarray(Train_Datas_1D)
    Test_x_1D = np.asarray(Test_Datas_1D)

    Train_x_1D = np.zeros((Train_x_1D.shape[0], Train_Datas_1D.shape[1], 1))
    for i in range(Train_x_1D.shape[0]):
        temp = np.resize(Train_x_1D[i], (Train_Datas_1D.shape[1], 1))
        Train_x_1D[i] = np.reshape(temp, (Train_Datas_1D.shape[1], 1))

    Test_x_1D = np.zeros((Test_x_1D.shape[0], Train_Datas_1D.shape[1], 1))
    for i in range(Test_x_1D.shape[0]):
        temp = np.resize(Test_x_1D[i], (Train_Datas_1D.shape[1], 1))
        Test_x_1D[i] = np.reshape(temp, (Train_Datas_1D.shape[1], 1))

    # Input layers
    input_1D = Input(shape=(Train_x_1D.shape[1], 1))  # 1D input (features)
    input_2D = Input(shape=(IMG_SIZE, IMG_SIZE, 3))  # 2D input (raw image)

    # 1D Convolution for features
    conv1d_out = Conv1D(filters=128, kernel_size=3, activation='relu')(input_1D)
    conv1d_out_flattened = Flatten()(conv1d_out)  # Flatten 1D convolution output

    # 2D Convolution for raw images
    conv2d_out = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(input_2D)

    # EfficientNetB7 Backbone
    efficient_net = EfficientNetB7(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')
    efficient_net.trainable = False  # Freeze the base model
    efficient_features = efficient_net(conv2d_out)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(efficient_features)

    # Concatenate features from 1D and 2D convolutions
    combined_features = Concatenate()([conv1d_out_flattened, gap])

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(combined_features)
    dense2 = Dense(int(sol[0]), activation='relu')(dense1)  # 128
    output_layer = Dense(Train_Targets.shape[1], activation='sigmoid')(dense2)

    # Build the model
    model = Model(inputs=[input_1D, input_2D], outputs=output_layer)

    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([Train_x_1D, Train_x_2D], Train_Targets, epochs=int(sol[1]), steps_per_epoch=int(sol[2]), batch_size=4,
              validation_data=([Test_x_1D, Test_x_2D], Test_Target))

    pred = model.predict([Test_x_1D, Test_x_2D])

    Eval = evaluate_error(pred, Test_Target)
    return Eval, pred

