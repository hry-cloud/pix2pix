# 加载，分割以及缩放maps的训练集图像
from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# 将目录中的全部图像加载到内存中去
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# 罗列目录中的文件名
	for filename in listdir(path):
		# 加载图像并修改图像尺寸
		pixels = load_img(path + filename, target_size=size)
		# 转换为numpy数组的形式
		pixels = img_to_array(pixels)
		# 分割为卫星图和地图
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# 数据集路径
path = 'maps/train/'
# 加载数据集
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# 保存为压缩的numpy数组
filename = 'maps_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
