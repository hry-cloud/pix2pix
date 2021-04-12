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
from keras.utils.vis_utils import plot_model

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
	# 权重初始化
	init = RandomNormal(stddev=0.02)
	# 添加上采样层
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# 添加批正则化
	g = BatchNormalization()(g, training=True)
	# 根据条件决定是否添加dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# 利用跳跃连接进行合并
	g = Concatenate()([g, skip_in])
	# relu激活
	g = Activation('relu')(g)
	return g

# 定义单独的生成器模型
def define_generator(image_shape=(256,256,3)):
	# 权重初始化
	init = RandomNormal(stddev=0.02)
	# 输入图像
	in_image = Input(shape=image_shape)
	# 编码器模型: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# 瓶颈层, 没有批正则化并且使用relu激活
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# 解码器模块: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
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

# 定义图像形状
image_shape = (256,256,3)
# 创建模型
model = define_generator(image_shape)
# 展示模型结构以及参数信息
model.summary()
# 模型可视化
plot_model(model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)
