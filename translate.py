# 加载模型并进行单张图像的转换
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import matplotlib

# 加载验证集的图像
def load_real_samples(filename):
    # 加载压缩数组
    data = load(filename)
    # 将数组分开，也即分为两张图像
    X1, X2 = data['arr_0'], data['arr_1']
    # 像素值从[0,255]转换为[-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

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

# 绘制源图像，不同的生成图像以及目标图像
def plot_images(src_img, gen_img1, gen_img2, tar_img):
	images = vstack((src_img, gen_img1, gen_img2, tar_img))
	# 将像素值从 [-1,1] 改为 [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'ED_Generated', 'Expected']
	# 一列一列的画图
	for i in range(len(images)):
		# 定义相关信息
		pyplot.subplot(1, 4, 1 + i)
		# 不显示轴
		pyplot.axis('off')
		# 作图
		pyplot.imshow(images[i])
		# 显示名称
		pyplot.title(titles[i])
	pyplot.savefig('rev_model_out.jpg')
	pyplot.show()

'''
# 加载源图像
src_image = load_image('des.jpg')
print('Loaded', src_image.shape)
# 加载模型
model = load_model('rev_model/rev_g_model_054800.h5')
# 从源图像生成图像
gen_image = model.predict(src_image)
# 将像素值范围从[-1,1]转换为[0,1]
gen_image = (gen_image + 1) / 2.0
# 保存并绘制图像
matplotlib.image.imsave('rev_out_5.jpg', gen_image[0])
pyplot.imshow(gen_image[0])
pyplot.axis('off')
pyplot.show()
'''

# 加载数据
[X2, X1] = load_real_samples('data/maps_val_256.npz')
print('Loaded', X1.shape, X2.shape)
# 加载模型
model1 = load_model('rev_70x70_model/new_rev_g_model_109600.h5')
model2 = load_model('rev_ED_model/rev_g_model_109600.h5')
#model3 = load_model('70x70_model/new_g_model_109600.h5')
#model4 = load_model('286x286_model/new_g_model_109600.h5')

# 随机选择样例
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# 利用源图像生成图像
gen_image1 = model1.predict(src_image)
gen_image2 = model2.predict(src_image)
#gen_image3 = model3.predict(src_image)
#gen_image4 = model4.predict(src_image)
# 绘制三张图像
plot_images(src_image, gen_image1, gen_image2, tar_image)
