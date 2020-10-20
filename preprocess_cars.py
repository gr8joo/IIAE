import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import h5py
import os
from PIL import Image

#reading v 7.3 mat file in python
#https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python

root_path = None
assert root_path is None
default_input_path= os.path.join(root_path, 'dataset/cars/raw/')
default_train_path= os.path.join(root_path, 'dataset/cars/train/')
default_test_path = os.path.join(root_path, 'dataset/cars/test/')

out_img_id = 0
for img_id in range(1, 192):
    input_path = default_input_path + 'car_' + str(img_id).zfill(3) + '_mesh.mat'
    if not os.path.exists(input_path):
        continue
    f = io.loadmat(input_path)
    for i in range(4):
        profile = f['im'][:,:,:, 0, i]
        for j in range(23):
            out_img_id += 1
            image = f['im'][:,:,:, j+1, i]
            output_image = np.concatenate([profile, image], axis=1)

            im = Image.fromarray(output_image)
            output_path = default_train_path + str(out_img_id).zfill(5) + '.jpg'
            im.save(output_path)

print('train_samples:', out_img_id)

out_img_id = 0
for img_id in range(192, 200):
    input_path = default_input_path + 'car_' + str(img_id).zfill(3) + '_mesh.mat'
    if not os.path.exists(input_path):
        continue
    f = io.loadmat(input_path)
    for i in range(4):
        profile = f['im'][:,:,:, 0, i]
        for j in range(23):
            out_img_id += 1
            image = f['im'][:,:,:, j+1, i]
            output_image = np.concatenate([profile, image], axis=1)

            im = Image.fromarray(output_image)
            output_path = default_test_path + str(out_img_id).zfill(3) + '.jpg'
            im.save(output_path)
    # f.close()

print('test_samples:', out_img_id)

#Convert image to uint8 (before saving as jpeg - jpeg doesn't support int16 format).
#Use simple linear conversion: subtract minimum, and divide by range.
#Note: the conversion is not optimal - you should find a better way.
#Multiply by 255 to set values in uint8 range [0, 255], and covert to type uint8.
'''
hi = np.max(image)
lo = np.min(image)
image = (((image - lo)/(hi-lo))*255).astype(np.uint8)
'''

'''
#Save as jpeg
#https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
im = Image.fromarray(image)
im.save(os.path.join(root_path, 'dataset/cars2/car_001_mesh.jpg'))

#Display image for testing
imgplot = plt.imshow(image)
plt.show()
'''