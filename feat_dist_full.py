import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import h5py
import os
from PIL import Image
import argparse

#reading v 7.3 mat file in python
#https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python

parser = argparse.ArgumentParser()

########## General Settings ##########
# Added features mode to extract features for retrieval
parser.add_argument("--root_path", required=True, help="path to folder containing train images")
parser.add_argument("--dataset_name", required=True, help="path to folder containing train images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
a = parser.parse_args()

if a.dataset_name.lower() == "mnist" or a.dataset_name.lower() == "mnist-cdcb":
    a.dataset_name = "MNIST-CDCB"
    total_num = 10000
elif a.dataset_name.lower() == "facades" or a.dataset_name.lower() == "facade":
    a.dataset_name = "facades"
    total_num = 106
elif a.dataset_name.lower() == "maps" or a.dataset_name.lower() == "map":
    a.dataset_name = "maps"
    total_num = 1098

X_SharedFeat = []
Y_SharedFeat = []
X_ExFeat = []
Y_ExFeat = []

X_SharedLogvar = []
Y_SharedLogvar = []
X_ExLogvar = []
Y_ExLogvar = []

Q_SharedFeat = []
Q_SharedLogvar = []

# Write down the path of test images and the number of samples.
default_test_path = a.output_dir

assert default_test_path != None
assert total_num != None

print('Loading features from: ', default_test_path)
for img_id in range(1, total_num+1):
    if a.dataset_name == "MNIST-CDCB":
        input_path = os.path.join(default_test_path, 'features', str(img_id).zfill(5) + '.mat')
    else:
        input_path = os.path.join(default_test_path, 'features', str(img_id) + '.mat')
    f = io.loadmat(input_path)

    X_SharedFeat.append(np.reshape(f['sR_X2Y'], [1,-1]))
    Y_SharedFeat.append(np.reshape(f['sR_Y2X'], [1,-1]))

    X_ExFeat.append(np.reshape(f['eR_X2Y'], [1,-1]))
    Y_ExFeat.append(np.reshape(f['eR_Y2X'], [1,-1]))


##### Representation Preprocess #####
X_SharedFeat = np.concatenate(X_SharedFeat, axis=0)
Y_SharedFeat = np.concatenate(Y_SharedFeat, axis=0)
X_SharedPositive = 0
Y_SharedPositive = 0

X_ExFeat = np.concatenate(X_ExFeat, axis=0)
Y_ExFeat = np.concatenate(Y_ExFeat, axis=0)
X_ExPositive = 0
Y_ExPositive = 0

XtoY_label = np.zeros((total_num,), dtype=np.int8)
XtoY_top3 = []
YtoX_label = np.zeros((total_num,), dtype=np.int8)
YtoX_top3 = []
for i in range(total_num):

    X_SharedFeat_repeat = np.repeat(np.array([X_SharedFeat[i]]), total_num, axis=0)
    X_SharedDiff = (X_SharedFeat_repeat - Y_SharedFeat) ** 2
    X_SharedDist = np.sum(X_SharedDiff, axis=-1)
    X_SharedDistMinIdx = np.argmin(X_SharedDist, axis=-1)
    XtoY_argsort = np.argsort(X_SharedDist, axis=-1)
    XtoY_top3.append(XtoY_argsort[:3])
    if X_SharedDistMinIdx == i:
        X_SharedPositive += 1
        XtoY_label[i] = 1

    Y_SharedFeat_repeat = np.repeat(np.array([Y_SharedFeat[i]]), total_num, axis=0)
    Y_SharedDiff = (Y_SharedFeat_repeat - X_SharedFeat) ** 2
    Y_SharedDist = np.sum(Y_SharedDiff, axis=-1)
    Y_SharedDistMinIdx = np.argmin(Y_SharedDist, axis=-1)
    YtoX_argsort = np.argsort(Y_SharedDist, axis=-1)
    YtoX_top3.append(YtoX_argsort[:3])
    if Y_SharedDistMinIdx == i:
        Y_SharedPositive += 1
        YtoX_label[i] = 1


    X_ExFeat_repeat = np.repeat(np.array([X_ExFeat[i]]), total_num, axis=0)
    X_ExDiff = (X_ExFeat_repeat - Y_ExFeat) ** 2
    X_ExDist = np.sum(X_ExDiff, axis=-1)
    X_ExDistMinIdx = np.argmin(X_ExDist, axis=-1)
    if X_ExDistMinIdx == i:
        X_ExPositive += 1

    Y_ExFeat_repeat = np.repeat(np.array([Y_ExFeat[i]]), total_num, axis=0)
    Y_ExDiff = (Y_ExFeat_repeat - X_ExFeat) ** 2
    Y_ExDist = np.sum(Y_ExDiff, axis=-1)
    Y_ExDistMinIdx = np.argmin(Y_ExDist, axis=-1)
    if Y_ExDistMinIdx == i:
        Y_ExPositive += 1

    print(i)
XtoY_top3 = np.array(XtoY_top3)+1
YtoX_top3 = np.array(YtoX_top3)+1
print("From:", default_test_path)
print("Shared_Accuracy X2Y (L2):", float(X_SharedPositive) / float(total_num), "               ("+str(X_SharedPositive)+")")
print("Shared_Accuracy Y2X (L2):", float(Y_SharedPositive) / float(total_num), "               ("+str(Y_SharedPositive)+")" )
print("Exclusive_Accuracy X2Y (L2):", float(X_ExPositive) / float(total_num), "               ("+str(X_ExPositive)+")" )
print("Exclusive_Accuracy Y2X (L2):", float(Y_ExPositive) / float(total_num), "               ("+str(Y_ExPositive)+")" )

# both_false = []
# for i in range(XtoY_label.shape[0]):
#     if XtoY_top3[i][0] != i+1 and YtoX_top3[i][0] != i+1:
#         both_false.append(i)
import pdb; pdb.set_trace()