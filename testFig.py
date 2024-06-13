import matplotlib.pyplot as plt
import numpy as np

# 创建一些示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建一个折线图
plt.plot(x, y)
# 自定义图像样式
plt.title('Sine Wave Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(['y = sin(x)'])
# 显示图像
plt.show()
# 保存图像，包括自定义样式
plt.savefig('custom_sine_wave.png')


