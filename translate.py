# 加载模型并进行单张图像的转换
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import expand_dims
from matplotlib import pyplot
import matplotlib

# 加载图像
def load_image(filename, size=(256,256)):
	# 加载具有指定尺寸大小的图像
	pixels = load_img(filename, target_size=size)
	# 转换为numpy数组
	pixels = img_to_array(pixels)
	# 将范围从[0,255]转换为[-1,1]
	pixels = (pixels - 127.5) / 127.5
	# 改变形状为一个样本
	pixels = expand_dims(pixels, 0)
	return pixels

# 加载源图像
src_image = load_image('des.jpg')
print('Loaded', src_image.shape)
# 加载模型
model = load_model('rev_model_087680.h5')
# 从源图像生成图像
gen_image = model.predict(src_image)
# 将像素值范围从[-1,1]转换为[0,1]
gen_image = (gen_image + 1) / 2.0
# 保存并绘制图像
matplotlib.image.imsave('rev_out.jpg', gen_image[0])
pyplot.imshow(gen_image[0])
pyplot.axis('off')
pyplot.show()
