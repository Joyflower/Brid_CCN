import itertools
import os
import subprocess
import time

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix

# 精确度（Precision)被定义为0.0 则过滤掉这些警告
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

epochs = 30 #初始化训练集为30
# 1.数据准备
# 读取每个文件中的图片
start_time = time.time() #开始计时

while True:
    file_path = f"D:\\PyProject\\Brid_GCN\\updateLabels\\plot2\\epoch{epochs}"
    os.makedirs(file_path, exist_ok=True)  # 调用系统的创建文件夹的方法
    imgs_path = glob.glob('test/test/*.jpg') #glob 模块来查找指定路径下的所有以 .jpg 结尾的文件
    #todo 等会尝试使用train进行训练 再使用test进行测试 看运行结果怎么样 报错
    #实在不行再把这个图片的路径给改一下

    # imgs_path = [img_p for img_p in imgs_path if '\\' in img_p and '.' in img_p]
    # all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]

    print("imgs_path" , imgs_path)

    # 获取标签的名称
    all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]
    #路径: imgs_path:test/test\\0000.jpg 先用\\进行分成两部分 再对第二部分的0000.jpg进行分割 提炼出第二部分jpg
    #这里由于处理的都是图片jpg,而不是前面的编号0000,所以统一取出后面的部分,并进行统一的编号进行训练,确保数据的一致性.
    # all_labels_name = [img_p.split('\\')[1].split('.')[0] for img_p in imgs_path] #取出0000,0001,0002作为编号
    print("all_labels_name:",all_labels_name)
    # 把标签名称进行去重
    labels_names = np.unique(all_labels_name)
    print("labels_names: " , labels_names)
    # 包装为字典，将名称映射为序号
    label_to_index = {name: i for i, name in enumerate(labels_names)}
    print("label_to_index: ",label_to_index)
    # 反转字典
    index_to_label = {v: k for k, v in label_to_index.items()}
    print("index_to_label: " , index_to_label)
    # 将所有标签映射为序号
    all_labels = [label_to_index[name] for name in all_labels_name]
    print("all_labels: " , all_labels)

    # 将数据随机打乱，划分为训练数据和测试数据
    np.random.seed(2023)
    # 设置种子确保随机数序列是确定的，可以在不同的运行中产生相同的结果
    random_index = np.random.permutation(len(imgs_path))
    # permutation函数生成一个随机的索引序列，该序列的长度与数据集中样本的数量相同,将数据随机打乱
    print("random_index: " , random_index)
    imgs_path = np.array(imgs_path)[random_index]
    # 将原始数据集中的图片路径按照随机索引重新排序，实现数据的随机打乱。
    print("imgs_path: " , imgs_path)
    all_labels = np.array(all_labels)[random_index]
    # 将原始数据集中的标签按照相同的随机索引重新排序，确保图片路径和对应的标签保持一致。
    print("all_labels: " , all_labels)

    # 切片，取90%作为训练数据，10%作为测试数据
    # 计算出90%数据对应的索引位置，即将总样本数乘以0.9，然后取整得到索引位置i。
    i = int(len(imgs_path) * 0.9)
    # 根据计算得到的索引位置，将前90%的数据样本路径和标签切片出来，作为训练数据集。
    train_path = imgs_path[:i]
    train_labels = all_labels[:i]
    # 剩余的10%数据样本路径和标签则切片出来，作为测试数据集。
    test_path = imgs_path[i:]
    test_labels = all_labels[i:]

    # 2.数据集构建与预处理
    # 构建数据集
    # 由于需要用到tensorflow 这里要调用from_tensor_slices()函数,传入两个切片: 一个是处理的图像 一个是对应的索引
    # 得到处理过后的DataSet数据集 每个元素为一个切片(元组) 第一个元素为图像 第二个元素为对应的索引
    train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))

    # 读取图片路径并进行图片预处理 用函数复用
    def load_img(path, label):
        image = tf.io.read_file(path) #读取指定路径下的文件内容
        image = tf.image.decode_jpeg(image, channels=3) # 函数用于将PEG编码的图像数据解码为张量,这里有3个通道,即R、G、B
        image = tf.image.resize(image, [256, 256])#调整图像大小为256x256像素
        #这里设置为256,256可以满足模型输入要求、提高数据统一性、提升计算效率，并保留大部分图像信息。
        image = tf.cast(image, tf.float32) #tf要求卷积神经网络的张量为浮点数,精度更高,使得图像处理更兼容。
        image = image / 255 #除以图像的最大像素值进行图像数值归一化 提高泛化能力
        return image, label

    # 自动选择最佳的并行处理线程数
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # 运用tf框架的并行处理深度学习机制
    # 对训练数据和测试数据应用预处理函数和并行处理
    train_ds = train_ds.map(load_img, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(load_img, num_parallel_calls=AUTOTUNE)
    # 结合机器硬件资源设置批量大小和缓冲区大小
    BATCH_SIZE = 24 #适中的训练批次
    BUFFER_SIZE = 1000 #设置缓冲区,用于后续充分打乱数据

    # 打乱、分批次和预取数据 shuffle打乱数据 batch分批次 prefetch预取数据
    # 提高模型的泛化能力、鲁棒性、消除顺序逆差。
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    input_shape = (256, 256, 3) # 定义输入图片形状的规格大小
    # 构建模型,通过堆叠不同类型的层来逐步提取输入图像的特征
    model = tf.keras.Sequential([
        #第一层
        #按顺序进行CNN各层的堆叠
        tf.keras.layers.Input(shape=input_shape),  # 使用Input(shape)作为输入层
        #构建卷积层:指定卷积核(过滤器)数量为64,卷积核大小为3*3像素区域,较好地捕捉图像特征,并指定激活函数relu(非线性)处理复杂图像效果优
        tf.keras.layers.Conv2D(64, (3, 3),  activation='relu'),
        #批量归一化层，用于加速模型训练过程并提高模型的泛化能力。
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        #最大池化层,用于降低特征图像的空间维度。
        tf.keras.layers.MaxPooling2D(),

        #第二层
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        #第三层
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        #第四层
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        #第五层
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        #第六层
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        #全局平均池化层，用于将每个特征图的所有元素求平均值
        tf.keras.layers.GlobalAveragePooling2D(),
        #全连接层，包含256个神经元，调用用ReLU激活函数。
        #这一层将上一层的所有输入连接到256个神经元
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        #全连接层，包含200个神经元，调用softmax激活函数，处理图像的多分类问题。
        #识别200多个类别的鸟类 神经元为200
        tf.keras.layers.Dense(200, activation='softmax')
    ])

    # 输出模型的摘要信息
    model.summary()

    # 编译模型
    model.compile(
        #指定模型训练的优化器 采用效果较好的Adam优化器(自适应算法)，并设置学习率为0.001(官网参数)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

        #指定损失函数,常用稀疏分类交叉熵损失函数
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        #确定评估指标为准确率accuracy,用于检测模型训练后的分类效果
        metrics=['accuracy'])

    # 定义保存模型的回调函数
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/best_model.h5.keras', #官方新写法: best_model.h5.keras 需要加上一个后缀.keras
        # save_weights_only=True,
        save_best_only=True,#只保存在验证集上性能最好的模型
        monitor='val_accuracy',#指定监视的指标为: 验证集上的准确率。
        mode='max', #指定指标模式,最大化验证准确率，设置为 max
        verbose=1  #输出一些保存模型时的提示信息
    )

    # 训练模型
    history = model.fit(
        train_ds, #传入训练集
        epochs=epochs, #设置epochs训练批次为epochs
        validation_data=test_ds, #指定验证数据
        callbacks=[checkpoint_callback],#在每次验证准确率提升时，会将模型(权重)保存下来。
    )

    # 绘制准确率曲线 通过曲线对比验证集和训练集上的分类准确率的收敛能力和拟合度 以此验证模型验证图像的泛化能力
    # 以训练批次作为横坐标 accuracy为模型训练集上的分类准确率
    plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    # 以训练批次作为横坐标 val_accuracy为模型验证集上的分类准确率
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    # 构建要保存的文件路径
    accuracy_path = os.path.join(file_path, 'accuracy.png')  # 将文件夹的路径进行拼接
    # print(file_path)
    plt.savefig(accuracy_path)  # 将图片保存到这里
    plt.show()

    #绘制损失曲线
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    # 构建要保存的文件路径
    loss_path = os.path.join(file_path, 'loss.png')  # 将文件夹的路径进行拼接
    # print(file_path)
    plt.savefig(loss_path)  # 将图片保存到这里
    plt.show()

    # 预测测试数据
    y_pred = model.predict(test_ds)
    # 将预测结果转换为类别序号
    y_pred = np.argmax(y_pred, axis=1)

    # 输出分类报告
    # 打印出分类报告，包括每个类别的精确度、召回率、f1分数以及支持度
    print(classification_report(test_labels, y_pred, target_names=labels_names))

    # 计算分类报告
    # report = classification_report(test_labels, y_pred, target_names=labels_names, output_dict=True)
    # report_df = pd.DataFrame(report).transpose()

    #确保报告中包含了所有在测试数据和预测结果中出现过的类别。
    report = classification_report(test_labels, #测试数据集的真实标签
                                   y_pred, #模型的预测结果
                                   target_names=labels_names, #用于分类报告中各个类别的标识
                                   output_dict=True, #存成字典
                                   labels=np.unique(np.concatenate([test_labels, y_pred])#将真实标签和预测标签进行拼接，然后使用np.unique函数获取唯一的标签值。
                            ))

    # debug前的代码 正常
    # report = classification_report(test_labels, y_pred, target_names=labels_names, output_dict=True,
    #                                labels=np.unique(test_labels))

    #将分类报告的字典转换为 pandas 的 DataFrame，并对行和列进行转置
    report_df = pd.DataFrame(report).transpose()
    # 将分类报告写入 Excel 文件
    report_df.to_excel("classification_report.xlsx", sheet_name="Predict Report")

    # 计算混淆矩阵
    # 混淆矩阵显示了模型的预测结果与真实标签之间的关系，从而可以看出模型在每个类别上的表现。
    cm = confusion_matrix(test_labels, y_pred, labels=np.arange(len(labels_names)))
    # cm = confusion_matrix(y_true, y_pred, labels=labels_names)
    # 显示混淆矩阵
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    # 生成刻度标签，用于设置坐标轴的刻度。
    tick_marks = np.arange(len(labels_names))
    plt.xticks(tick_marks, labels_names, rotation=45)
    #设置x轴的刻度标签 将类别名称显示在x坐标轴上,并进行了 45 度的旋转,以避免重叠。
    plt.yticks(tick_marks, labels_names)
    # 设置y轴的刻度标签 将类别名称显示在y坐标轴上

    thresh = cm.max() / 2. #计算混淆矩阵中最大值的一半，设置为阈值，用于决定文本标签的颜色
    # 直观地分析混淆矩阵中的预测结果，进而评估模型在不同类别上的性能表现。
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 遍历混淆矩阵的所有行列索引组合。这样可以确保在每个单元格中都添加文本标签。
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", #文本水平居中对齐
                 color="white" if cm[i, j] > thresh else "black" #如果混淆矩阵中的数值大于阈值，文本颜色将被设置为白色，否则设置为黑色。
                 )
    plt.tight_layout() #调整图像合适间距
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 构建要保存的文件路径
    Confusion_Matrix_path = os.path.join(file_path, 'Confusion_Matrix.png')  # 将文件夹的路径进行拼接
    # print(file_path)
    plt.savefig(Confusion_Matrix_path)  # 将图片保存到这里
    plt.show()

    end_time = time.time() # 结束计时

    # 计算执行时间 用于预算模型训练的时间
    execution_time = end_time - start_time
    print("执行时间为: {:.5f} 秒".format(execution_time))

    #主线程结束后,进入子线程。
    subprocess.run(["python", "updateLabelsTest.py","--file_path", file_path])

    #模型的最大训练集为100
    if epochs >= 30:
        break
    else:
        epochs += 10

    time.sleep(300) #机器休息3分钟
