import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from model import Attention_ResUNet
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.losses import categorical_crossentropy
import tensorflow as tf

# custom data generators
def load_img(img_dir, img_list):
    images = []

    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1]=='npy'):
            image=np.load(img_dir+image_name)
            images.append(image)

    images=np.array(images)
    return (images)
def imageLoader(img_dir, img_list, mask_dir, mask_list,batch_size):
    L=len(img_list)     # 258

    while True:
        batch_start=0
        batch_end=batch_size

        while batch_start <L:
            limit=min(batch_end,L)

            X=load_img(img_dir, img_list[batch_start:limit])
            Y=load_img(mask_dir, mask_list[batch_start:limit])

            yield (X,Y)
            batch_start += batch_size
            batch_end += batch_size


# compile models
# Cat model selected for training
def compile_model_cat(model):
    learning_rate = 0.0001
    optm = keras.optimizers.Adam(learning_rate)
    # model.compile(optimizer=optm, loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def compile_model_sgd(model):
    learning_rate=0.0001
    opt=keras.optimizers.SGD(lr=learning_rate,momentum=0.9,decay=0.0005)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model

# ploting history after training
def plot_history_loss(history):
    plt.subplot(1,1,1)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()
def plot_history_accuracy(history):
    plt.subplot(1,1,1)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.show()


# main function
if __name__ == "__main__":
    train_img_dir = 'D:\\Ishaq\\b15\\mix\\train_val_test\\train\\images\\'
    train_mask_dir = 'D:\\Ishaq\\b15\\mix\\train_val_test\\train\\masks\\'
    train_img_list = os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)

    val_img_dir = 'D:\\Ishaq\\b15\\mix\\train_val_test\\val\\images\\'
    val_mask_dir = 'D:\\Ishaq\\b15\\mix\\train_val_test\\val\\masks\\'
    val_img_list = os.listdir(val_img_dir)
    val_mask_list = os.listdir(val_mask_dir)

    batch_size = 10

    train_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
    val_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

    steps_per_epoch = len(train_img_list) // batch_size
    val_steps_per_epoch = len(val_img_list) // batch_size

    input_shape=(240,240,4)

    model=Attention_ResUNet(input_shape,NUM_CLASSES=5,dropout_rate=0.2, batch_norm=True)

    model=compile_model_cat(model)

#    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=25, verbose=1),
#                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.9,patience=10,min_lr=0.00001,verbose=1),
#                 keras.callbacks.ModelCheckpoint("Attention_ResUNEt_epochs_200_cat_2015.hdf5",verbose=1,save_best_only=True,save_weights_only=True),
#                 keras.callbacks.CSVLogger('Attention_ResUNEt_epochs_200_cat_2015.csv', separator=',', append=False)]

    callbacks1 = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.00001,verbose=1),
                 keras.callbacks.ModelCheckpoint("Attention_ResUNEt_epochs_150_cat_2015.hdf5", verbose=1,save_best_only=True, save_weights_only=True),
                 keras.callbacks.CSVLogger('Attention_ResUNEt_epochs_150_cat_2015.csv', separator=',', append=False)]
 #callbacks2 = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2,min_delta=1E-7, verbose=1),
 #keras.callbacks.CSVLogger('Attention_ResUNEt_epochs_10_cat_2015.csv', separator=',', append=False)]

    history = model.fit(train_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=10,
                        verbose=1,
                        validation_data=val_datagen,
                        validation_steps=val_steps_per_epoch,
                        callbacks=[callbacks1])

    model.save('Attention_ResUNEt_epochs_150_cat_2015.hdf5')
    plot_history_loss(history)
    plot_history_accuracy(history)

    # from keras.models import load_model
    # my_model=load_model('Attention_ResUNEt_epochs_200_cat_2015.hdf5')
    # history = my_model.fit(train_datagen,
    #                     steps_per_epoch=steps_per_epoch,
    #                     epochs=100,
    #                     verbose=1,
    #                     validation_data=val_datagen,
    #                     validation_steps=val_steps_per_epoch,
    #                     callbacks=[callbacks1])
    # my_model.save('Attention_ResUNEt_epochs_200a_cat_2015.hdf5')
    # plot_history_loss(history)
    # plot_history_accuracy(history)