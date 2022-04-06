import cv2
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


# function used in both times
def normalize(x):
    if(np.sum(x)!=0):
        x=(x-np.min(x))/(np.max(x)-np.min(x))
    return x
def show_io(image,mask):
    plt.figure(figsize=(8, 8))
    plt.subplot(231)
    plt.imshow(image[:, :, 0])
    plt.title('Image flair')
    plt.subplot(232)
    plt.imshow(image[:, :, 1])
    plt.title('Image t1')
    plt.subplot(233)
    plt.imshow(image[:, :, 2])
    plt.title('Image t1ce')
    plt.subplot(234)
    plt.imshow(image[:, :, 0])
    plt.title('Image t2')
    plt.subplot(235)
    mask=np.argmax(mask,axis=2)
    plt.imshow(mask[:, :])
    plt.title('Mask')
    plt.show()


# function not used (only experiments)
def scalar_normalization(x):
    scaler = MinMaxScaler()
    scaler.fit(x)
    x=scaler.transform(x)
    return x
def make_numpy_files_test():
    f, t1, t1ce, t2, gt = read_files()
    count=0
    for i in range(220):
        img_f = sitk.GetArrayFromImage(sitk.ReadImage(f[i]))
        img_t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        img_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce[i]))
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt[i]))
        for j in range(155):
            s_f=img_f[j,29:221,20:212]
            s_t1=img_t1[j,29:221,20:212]
            s_t1ce=img_t2[j,29:221,20:212]
            s_t2=img_t2[j,29:221,20:212]
            s_gt=img_gt[j,29:221,20:212]
            # print(s_f.shape,s_t1.shape,s_t1ce.shape,s_t2.shape,s_gt.shape)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1))
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce))
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2))
            s_f=normalize(s_f)
            s_t1=normalize(s_t1)
            s_t1ce=normalize(s_t1ce)
            s_t2=normalize(s_t2)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1))
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce))
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2))
            image=np.stack([s_f,s_t1,s_t1ce,s_t2], axis=2)
            image=image.astype(np.float32)
            mask=to_categorical(s_gt,num_classes=5)
            # print(image.shape,image.dtype,np.min(image), np.max(image))
            # print(mask.shape,mask.dtype, np.min(mask), np.max(mask))
            if((np.sum(s_gt))!=0):
                print(i, j, count)
                # show_io(image,mask)
                np.save('D:\\Ishaq\\b15\\numpy_data\\images\\image_'+str(count) + '.npy',image)
                np.save('D:\\Ishaq\\b15\\numpy_data\\masks\\mask_' + str(count) + '.npy',mask)
                count=count+1


# function used for first model_192_192 training
def read_files():
    train_dirs = "D:\\Ishaq\\b15\\HGG\\**"
    patient_list = glob.glob(train_dirs)
    files_t1 = []
    files_t1c = []
    files_t2 = []
    files_flair = []
    files_labels = []
    for patient in patient_list:
        patient_files = glob.glob(patient + "\\**\\*.mha")
        for file in patient_files:
            if "MR_T1." in file:
                files_t1.append(file)
            elif "MR_T1c." in file:
                files_t1c.append(file)
            elif "MR_T2." in file:
                files_t2.append(file)
            elif "MR_Flair." in file:
                files_flair.append(file)
            elif "OT." in file:
                files_labels.append(file)
    print(len(files_t1), len(files_t1c), len(files_t2), len(files_flair), len(files_labels))
    return files_flair, files_t1, files_t1c, files_t2, files_labels
def make_numpy_files():
    f, t1, t1ce, t2, gt = read_files()
    count=0
    for i in range(220):
        img_f = sitk.GetArrayFromImage(sitk.ReadImage(f[i]))
        img_t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        img_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce[i]))
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt[i]))
        for j in range(155):
            s_f=img_f[j,29:221,20:212]
            s_t1=img_t1[j,29:221,20:212]
            s_t1ce=img_t2[j,29:221,20:212]
            s_t2=img_t2[j,29:221,20:212]
            s_gt=img_gt[j,29:221,20:212]
            # print(s_f.shape,s_t1.shape,s_t1ce.shape,s_t2.shape,s_gt.shape)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1))
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce))
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2))
            s_f=normalize(s_f)
            s_t1=normalize(s_t1)
            s_t1ce=normalize(s_t1ce)
            s_t2=normalize(s_t2)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1))
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce))
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2))
            image=np.stack([s_f,s_t1,s_t1ce,s_t2], axis=2)
            image=image.astype(np.float32)
            s_gt=s_gt.astype(np.uint8)
            mask=to_categorical(s_gt,num_classes=5)
            # print(image.shape,image.dtype,np.min(image), np.max(image))
            # print(mask.shape,mask.dtype, np.min(mask), np.max(mask))
            if((np.sum(s_gt))!=0):
                print(i, j, count)
                # show_io(image,mask)
                np.save('D:\\Ishaq\\b15\\numpy_data\\images\\image_'+str(count) + '.npy',image)
                np.save('D:\\Ishaq\\b15\\numpy_data\\masks\\mask_' + str(count) + '.npy',mask)
                count=count+1

def train_val_test_split():
    import splitfolders
    # input_folder = 'D:\\Ishaq\\b15\\numpy_data\\'
    input_folder = 'D:\\Ishaq\\b15\\mix\\lgg_hgg'
    output_folder= 'D:\\Ishaq\\b15\\mix\\train_val_test\\'
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.80, .10, .10),group_prefix=None)
# checkpoint=keras.callbacks.ModelCheckpoint("LGG_full_10.hdf5",monitor='val_loss',save_best_only=True)
# data preprocessing for second model_240_240 training
# callbacks=[keras.callbacks.EarlyStopping(patience=10,verbose=1),
#                keras.callbacks.ReduceLROnPlateau(factor=0.1,patience=5,min_lr=0.00001,verbose=1),
#                keras.callbacks.ModelCheckpoint("xyz.hdf5",verbose=1,save_best_only=True,save_weights_only=True)]
def read_files_lgg():
    train_dirs = "D:\\Ishaq\\b15\\LGG\\**"
    patient_list = glob.glob(train_dirs)
    files_t1 = []
    files_t1c = []
    files_t2 = []
    files_flair = []
    files_labels = []
    for patient in patient_list:
        patient_files = glob.glob(patient + "\\**\\*.mha")
        for file in patient_files:
            if "MR_T1." in file:
                files_t1.append(file)
            elif "MR_T1c." in file:
                files_t1c.append(file)
            elif "MR_T2." in file:
                files_t2.append(file)
            elif "MR_Flair." in file:
                files_flair.append(file)
            elif "OT." in file:
                files_labels.append(file)
    print(len(files_t1), len(files_t1c), len(files_t2), len(files_flair), len(files_labels))
    return files_flair, files_t1, files_t1c, files_t2, files_labels
def read_files_hgg():
    train_dirs = "D:\\Ishaq\\b15\\HGG\\**"
    patient_list = glob.glob(train_dirs)
    files_t1 = []
    files_t1c = []
    files_t2 = []
    files_flair = []
    files_labels = []
    for patient in patient_list:
        patient_files = glob.glob(patient + "\\**\\*.mha")
        for file in patient_files:
            if "MR_T1." in file:
                files_t1.append(file)
            elif "MR_T1c." in file:
                files_t1c.append(file)
            elif "MR_T2." in file:
                files_t2.append(file)
            elif "MR_Flair." in file:
                files_flair.append(file)
            elif "OT." in file:
                files_labels.append(file)
    print(len(files_t1), len(files_t1c), len(files_t2), len(files_flair), len(files_labels))
    return files_flair, files_t1, files_t1c, files_t2, files_labels

def make_numy_lgg():
    f, t1, t1ce, t2, gt = read_files_lgg()
    count = 30036
    for i in range(54):
        print(i)
        img_f = sitk.GetArrayFromImage(sitk.ReadImage(f[i]))
        img_t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        img_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce[i]))
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt[i]))
        for j in range(155):
            s_f = img_f[j,:,:]
            s_t1 = img_t1[j,:,:]
            s_t1ce =img_t2[j,:,:]
            s_t2 = img_t2[j,:,:]
            s_gt = img_gt[j,:,:]
            # print(s_f.shape,s_t1.shape,s_t1ce.shape,s_t2.shape,s_gt.shape)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1),s_t1.dtype)
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce),s_t1ce.dtype)
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2),s_t2.dtype)
            s_f = normalize(s_f)
            s_t1 = normalize(s_t1)
            s_t1ce = normalize(s_t1ce)
            s_t2 = normalize(s_t2)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1),s_t1.dtype)
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce),s_t1ce.dtype)
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2),s_t2.dtype)
            image = np.stack([s_f, s_t1, s_t1ce, s_t2], axis=2)
            image = image.astype(np.float32)
            s_gt = s_gt.astype(np.uint8)
            mask = to_categorical(s_gt, num_classes=5)
            # print(image.shape,image.dtype,np.min(image), np.max(image))
            # print(mask.shape,mask.dtype, np.min(mask), np.max(mask))
            if ((np.sum(image)) != 0):
                # show_io(image,mask)
                np.save('D:\\Ishaq\\b15\\numpy_data\\lgg\\images\\image_' + str(count) + '.npy', image)
                np.save('D:\\Ishaq\\b15\\numpy_data\\lgg\\masks\\mask_' + str(count) + '.npy', mask)
                count = count + 1
def make_numy_hgg():
    f, t1, t1ce, t2, gt = read_files_hgg()
    count = 0
    for i in range(220):
        print(i)
        img_f = sitk.GetArrayFromImage(sitk.ReadImage(f[i]))
        img_t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1[i]))
        img_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(t1ce[i]))
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2[i]))
        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt[i]))
        for j in range(155):
            s_f = img_f[j,:,:]
            s_t1 = img_t1[j,:,:]
            s_t1ce =img_t2[j,:,:]
            s_t2 = img_t2[j,:,:]
            s_gt = img_gt[j,:,:]
            # print(s_f.shape,s_t1.shape,s_t1ce.shape,s_t2.shape,s_gt.shape)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1),s_t1.dtype)
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce),s_t1ce.dtype)
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2),s_t2.dtype)
            s_f = normalize(s_f)
            s_t1 = normalize(s_t1)
            s_t1ce = normalize(s_t1ce)
            s_t2 = normalize(s_t2)
            # print("f_max=", np.max(s_f), " f_min=", np.min(s_f),s_f.dtype)
            # print("t1_max=", np.max(s_t1), " t1_min=", np.min(s_t1),s_t1.dtype)
            # print("t1ce_max=", np.max(s_t1ce), " t1ce_min=", np.min(s_t1ce),s_t1ce.dtype)
            # print("t2_max=", np.max(s_t2), " t2_min=", np.min(s_t2),s_t2.dtype)
            image = np.stack([s_f, s_t1, s_t1ce, s_t2], axis=2)
            image = image.astype(np.float32)
            s_gt = s_gt.astype(np.uint8)
            mask = to_categorical(s_gt, num_classes=5)
            # print(image.shape,image.dtype,np.min(image), np.max(image))
            # print(mask.shape,mask.dtype, np.min(mask), np.max(mask))
            if ((np.sum(image)) != 0):
                # show_io(image,mask)
                np.save('D:\\Ishaq\\b15\\numpy_data\\hgg\\images\\image_' + str(count) + '.npy', image)
                np.save('D:\\Ishaq\\b15\\numpy_data\\hgg\\masks\\mask_' + str(count) + '.npy', mask)
                count = count + 1