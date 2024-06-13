
import os
import matplotlib.pyplot as plt


# 创建一个名为"epoch_folder"的文件夹，epoch_value是当前的循环变量的值
epoch_value = 80
file_path = f"D:\\PyProject\\Brid_GCN\\updateEpoches\\plot\\epoch{epoch_value}"
os.makedirs(file_path, exist_ok=True) #调用系统的创建文件夹的方法

# 构建要保存的文件路径
label_trends_path = os.path.join(file_path, 'label_trends.png')#将文件夹的路径进行拼接
print(label_trends_path)
# print(file_path)
plt.savefig(label_trends_path) #将图片保存到这里

# 构建要保存的文件路径
label_labels_path = os.path.join(file_path, 'label_labels.png')#将文件夹的路径进行拼接
print(label_labels_path)
# print(file_path)
plt.savefig(label_labels_path) #将图片保存到这里




