import shutil

# source_path = '/ext2/hjhwang/dataset/MNIST-CDCB/train/'
# target_path = '/ext2/hjhwang/dataset/MNIST-CDCB/valid/'
source_path = '/ext2/hjhwang/dataset/maps/train/'
target_path = '/ext2/hjhwang/dataset/maps/valid/'

# Copy 800 images for validation from training set
for i in range(1,801):
    # source_img = source_path + str(i).zfill(5) + '.png'
    # target_img = target_path + str(i).zfill(5) + '.png'
    source_img = source_path + str(i) + '.jpg'
    target_img = target_path + str(i) + '.jpg'
    shutil.copyfile(source_img, target_img)
