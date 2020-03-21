import random
import os
train_files = open("train_files.txt", "w")
val_files = open("val_files.txt", "w")
train_list = []
val_list = []
train_index = []
val_index = []
imindex = 0
for i in range(7481):
    count = 0
    for j in range(1, 4):
        if os.path.exists("/home/afayou/Documents/data_object/training/prev_2/{:06d}_{:02d}.png".format(i, j)):
            count += 1
            if count is 3:
                train_index.append(imindex)
                imindex += 1
imindex = 0
for i in range(7518):
    count = 0
    for j in range(1, 4):
        if os.path.exists("/home/afayou/Documents/data_object/testing/prev_2/{:06d}_{:02d}.png".format(i, j)):
            count += 1
            if count is 3:
                val_index.append(imindex)
                imindex += 1
for i in train_index:
    item = ["training", str(i), "l"]
    train_list.append(item)
    # item = ["training", str(i), "r"]
    # train_list.append(item)
random.shuffle(train_list)
for i in val_index:
    item = ["validation", str(i), "l"]
    val_list.append(item)
    # item = ["testing", str(i), "r"]
    # val_list.append(item)
random.shuffle(val_list)
for i in train_list:
    train_files.writelines(" ".join(i) + "\n")
for i in val_list:
    val_files.writelines(" ".join(i) + "\n")
train_files.close()
val_files.close()

