# 将谷歌地图转换为卫星图
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import load_model
from matplotlib import pyplot

# 定义判别器模型
# 70x70 Patch GAN
def define_discriminator(image_shape):
	# 权重初始化
	init = RandomNormal(stddev=0.02)
	# 源图像输入
	in_src_image = Input(shape=image_shape)
	# 目标图像输入
	in_target_image = Input(shape=image_shape)
	# 将图像在通道尺度上连接
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# 倒数第二个输出层
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch输出
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# 定义模型
	model = Model([in_src_image, in_target_image], patch_out)
	# 编译模型
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# 定义编码器模块
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# 权重初始化
	init = RandomNormal(stddev=0.02)
	# 添加下采样层
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# 根据条件决定是否添加批正则化
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu激活
	g = LeakyReLU(alpha=0.2)(g)
	return g

# 定义解码器模块
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# 权重正则化
	init = RandomNormal(stddev=0.02)
	# 添加上采样层
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# 添加批正则化
	g = BatchNormalization()(g, training=True)
	# 根据条件决定是否添加dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# 添加跳跃连接
	g = Concatenate()([g, skip_in])
	# relu激活
	g = Activation('relu')(g)
	return g

# 定义独立的生成器模块
def define_generator(image_shape=(256,256,3)):
	# 权重初始化
	init = RandomNormal(stddev=0.02)
	# 输入图像
	in_image = Input(shape=image_shape)
	# 编码器模型
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# 瓶颈层，没有批正则化和relu激活
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# 解码器模型
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# 输出
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# 定义模型
	model = Model(in_image, out_image)
	return model

# 定义判别器和生成器的复合模型，用于更新生成器
def define_gan(g_model, d_model, image_shape):
	# 冻结判别器的参数
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# 定义源图像
	in_src = Input(shape=image_shape)
	# 将源图像作为生成器的输入
	gen_out = g_model(in_src)
	# 将源图像和生成器的输出作为判别器的输入
	dis_out = d_model([in_src, gen_out])
	# 将源图像作为输入，输出生成的图像以及分类结果
	model = Model(in_src, [dis_out, gen_out])
	# 编译模型
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# 加载并且准备用于训练的图像
def load_real_samples(filename):
	# 加载压缩的数组
	data = load(filename)
	# 将数组拆为两部分，也即对应源图像和目标图像
	X1, X2 = data['arr_0'], data['arr_1']
	# 将尺寸由[0,255]变为[-1,1]，也即归一化
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X2, X1]

# 从样本中随机选择一批图像，返回图像对以及对应的分类标签
def generate_real_samples(dataset, n_samples, patch_shape):
	# 将数据集拆分
	trainA, trainB = dataset
	# 随机选择样例
	ix = randint(0, trainA.shape[0], n_samples)
	# 恢复选择的图像
	X1, X2 = trainA[ix], trainB[ix]
	# 生成真实分类的标签（1）
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# 生成一批图像并且返回图像和对应分类标签
def generate_fake_samples(g_model, samples, patch_shape):
	# 生成（假）图像
	X = g_model.predict(samples)
	# 生成‘假’分类标签（0）
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# 生成样例并且保存为plot图的形式，同时保存模型
def summarize_performance(step, g_model, dataset, n_samples=3):
	# 选择一个输入图像的实例
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# 生成一批假样品
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# 将所有像素值范围从[-1,1]变为[0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# 绘制真实的源图像
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# 绘制生成的目标图像
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# 绘制真实的目标图像
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# 将绘制的图保存到文件
	filename1 = 'rev_70x70_model/new_rev_plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# 保存生成器模型
	filename2 = 'rev_70x70_model/new_rev_g_model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# 训练pix2pix模型
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	# 确定判别器输出的方形形状
	n_patch = d_model.output_shape[1]
	# 拆分数据集
	trainA, trainB = dataset
	# 计算每一个训练轮次的图像批数
	bat_per_epo = int(len(trainA) / n_batch)
	# 计算训练的迭代次数
	n_steps = bat_per_epo * n_epochs
	# 不同的训练轮次
	for i in range(n_steps):
		# 选择一批真实的样例
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# 生成一批虚假的样例
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# 利用真实样例来更新判别器
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# 利用生成样例来更新判别器
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# 更新生成器
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# 总结性能
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# 总结模型的性能
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)

# 加载图像数据
dataset = load_real_samples('data/maps_train_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# 基于加载的数据集来定义输入的形状
image_shape = dataset[0].shape[1:]
# 定义模型
load_from_checkpoint = False
d_model_file = "rev_d_modle_010960.h5"
g_model_file = "rev_g_modle_010960.h5"
if load_from_checkpoint:
	d_model = load_model(d_model_file)
	g_model = load_model(g_model_file)
else:
	d_model = define_discriminator(image_shape)
	g_model = define_generator(image_shape)
# 定义复合模型
gan_model = define_gan(g_model, d_model, image_shape)
# 训练模型
train(d_model, g_model, gan_model, dataset)

from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
# load all images in a directory into memory
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# dataset path
path = 'maps/train/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'maps_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
