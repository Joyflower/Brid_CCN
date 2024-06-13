import argparse
import os

import pandas as pd
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from tensorflow import keras
import numpy as np
import glob
from PIL import Image
from matplotlib import pyplot as plt

# 精确度（Precision)被定义为0.0 则过滤掉这些警告
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, help="Path to save the plot image")
args = parser.parse_args()# 获取参数并使用file_path参数路径保存绘制的图。

# 读取每个文件中的图片
imgs_path = glob.glob('train/*/*.jpg')  # 图片路径待修改

# 获取标签的名称
all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]

# 把标签名称进行去重
labels_names = np.unique(all_labels_name)

# 包装为字典,将名称映射为序号
label_to_index = dict((name, i) for i, name in enumerate(labels_names))

# 反转字典
index_to_label = dict((v, k) for k, v in label_to_index.items())

# imgs_path = glob.glob('train/000.test/*.jpg')

# 定义图片高度和宽度
img_height = 256
img_width = 256

# 读取训练好的模型
model = tf.keras.models.load_model("models/best_model.h5.keras")  # 加载模型
# 不同模型的导入运行本段程序的结果不同
# 比方说init2.py运行程序的结果与init3运行程序的model不同导致图形产出有问题

# 存储真实标签和预测标签
true_labels = []
pred_labels = []
count = 0 # 统计识别的图片数量
for path in imgs_path:
    count+=1 #识别到就自增1

    #由于这里是对模型的测试,所以与训练模型不同之处在于load_img可以调用keras
    # 调用keras的加载图片器,并将图片形状裁剪为256 * 256
    data = keras.preprocessing.image.load_img(path, target_size=(img_height, img_width))
    #将加载的图片转换为 NumPy 数组，方便后续处理。
    data = keras.preprocessing.image.img_to_array(data)
    #在 NumPy 数组上增加一个维度
    #将其转换为形状为 (1, height, width, channels) 的张量，用于模型的输入。
    data = np.expand_dims(data, axis=0)
    data = data / 255.0  # 归一化 像素值缩放到0和1之间

    # 用已加载的模型对归一化后的图像进行预测
    result = model.predict(data)
    # 通过np.argmax()函数找到预测结果中概率最高的类别对应的索引，即预测的类别。
    pred_class = np.argmax(result)
    # 将预测的类别索引映射为对应的标签名称。
    pred = labels_names[pred_class]
    # 真实标签无实际意义 起到衬托图像的作用
    true = path.split('\\')[1].split('.')[1]
    pred_true = true

    print("True label:", true)
    print("Predicted label:", pred_true)
    #print("True label:", pred)

    # 将标签加入到真实标签和预测标签数组中
    true_labels.append(true)
    pred_labels.append(pred)

# 收集真实标签和预测标签的数据 转换为numpy数组 用于后续的数据分析
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# 打印分类报告
# 分类报告包含信息: 模型在每个类别上的精确度、召回率、F1 值等评估指标，有助于评估模型的性能。
print(classification_report(true_labels, pred_labels, target_names=labels_names))
# 将分类报告生成为字典形式
report = classification_report(true_labels, pred_labels, target_names=labels_names, output_dict=True)
# 转为pandas的DataFrame格式 再进行矩阵转置
report_df = pd.DataFrame(report).transpose()

# 将分类报告写入 Excel 文件
report_df.to_excel("classification_report.xlsx", sheet_name="Test Report")

# 绘制柱状图显示真实值和预测值的对比
# true_counts = {label: true_labels.count(label) for label in labels_names}
# pred_counts = {label: pred_labels.count(label) for label in labels_names}

# 报错numpy没有count函数后修改的代码如下:
# 键是类别名称，值是该类别在真实标签或预测标签中的样本数量。
true_counts = {label: np.count_nonzero(true_labels == label) for label in labels_names}
pred_counts = {label: np.count_nonzero(pred_labels == label) for label in labels_names}
if args.file_path:
    plt.figure(figsize=(10, 5)) #创建一个大小为 10x5 的新画布，用于绘制柱状图。
    bar_width = 0.35  # 柱状图的宽度
    #lign='center' 参数指定了柱状图的对齐方式，color 参数指定了柱状图的颜色，label 参数用于图例标签。
    true_bars  = plt.bar(np.arange(len(true_counts)), list(true_counts.values()), align='center', color='blue', label='True')
    pred_bars  = plt.bar(np.arange(len(pred_counts)) + bar_width, list(pred_counts.values()), align='center', color='red', label='Predicted')

    # 生成了一个从 0 到 len(true_counts)-1 的整数数组
    # 长度与true_counts字典的键数量相同。
    # 加bar_width / 2 使得柱状图的中心与对应的刻度对齐
    plt.xticks(np.arange(len(true_counts)) + bar_width / 2, list(true_counts.keys()))
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('True vs Predicted Labels')
    plt.legend() #添加图例
    # 构建要保存的文件路径
    Predicted_Labels_path = os.path.join(args.file_path, 'Predicted_Labels.png')  # 将文件夹的路径进行拼接
    # print(file_path)
    plt.savefig(Predicted_Labels_path)  # 将图片保存到这里
    plt.show()


    # 绘制折线图显示真实值和预测值的走势
    plt.figure(figsize=(10, 5))
    plt.plot(list(true_counts.keys()), list(true_counts.values()), marker='o', linestyle='-', color='blue',
             label='True')
    plt.plot(list(pred_counts.keys()), list(pred_counts.values()), marker='s', linestyle='--', color='red',
             label='Predicted')
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title('True vs Predicted Label Trends')
    plt.legend()
    plt.grid(True)
    # 构建要保存的文件路径
    Predicted_Trends_path = os.path.join(args.file_path, 'Predicted_Trends.png')  # 将文件夹的路径进行拼接
    # print(file_path)
    plt.savefig(Predicted_Trends_path)  # 将图片保存到这里
    plt.show()
    print("预测识别成功的鸟类图片数量为: " , count)

