# Example, Test
from dataloader import NSD_Dataset
import torch
import time

# Default
data_dir = 'C:/Users/ljh/Desktop/NSD/data'
subject_num = "01"
batch_size = 16

# NSD Dataset
train_dataset = NSD_Dataset(data_dir, subject_num, "train") # dataset_type
val_dataset = NSD_Dataset(data_dir, subject_num, "val")
test_dataset = NSD_Dataset(data_dir, subject_num, "test")
# train_ae_dataset = NSD_Dataset(data_dir, subject_num, "train_ae")


# DataLoader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Test one batch
# Train
batch = next(iter(train_dataloader))
images = batch['Image']
fmri_data = batch['fMRI']
image_indices = batch['Image_Index']

# Print
for i in range(len(images)):
    print("Image: {}".format(images[i]))
    print("Image_Index: {}".format(image_indices[i]))
    print("fMRI:")
    for key, value in fmri_data.items():
        print("- {}: {}".format(key, value[i]))
        print(value[i].shape)
    print()

# Test one batch
# Val
time.sleep(2)
batch = next(iter(val_dataloader))
images = batch['Image']
fmri_data = batch['fMRI']
image_indices = batch['Image_Index']

# Print
for i in range(len(images)):
    print("Image: {}".format(images[i]))
    print()
    print("Image Shape {}".format(images[i].shape))
    print()
    print("Image_Index: {}".format(image_indices[i]))
    print("fMRI:")
    for key, value in fmri_data.items():
        print("- {}: {}".format(key, value[i]))
        print(value[i].shape)
    print()

# Test one batch
# Test
time.sleep(2)
batch = next(iter(test_dataloader))
images = batch['Image']
fmri_data = batch['fMRI']
image_indices = batch['Image_Index']

# Print
for i in range(len(images)):
    print("Image: {}".format(images[i]))
    print("Image_Index: {}".format(image_indices[i]))
    print("fMRI:")
    for key, value in fmri_data.items():
        print("- {}: {}".format(key, value[i]))
        print(value[i].shape)
    print()