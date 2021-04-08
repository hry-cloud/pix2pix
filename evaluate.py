import numpy as np
import math
import skimage
from PIL import Image
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint

# 计算峰值信噪比PSNR
def PSNR(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2)
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)

# 计算结构相似性SSIM
def SSIM(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*255)
    c2 = np.square(0.03*255)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

# 计算语义分割的各个指标
"""
混淆矩阵横列代表预测值，竖列代表真实值
confusionMetric  
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    # 返回全部类别的整体像素准确率
    # PA = acc = (TP + TN) / (TP + TN + FP + TN)
    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    # 返回每个类别的像素准确率
    # acc = (TP) / TP + FP
    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        meanAcc = np.nanmean(classAcc)
        # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
        return meanAcc

    # 交集为TP，并集为TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    def meanIntersectionOverUnion(self):
        # 取对角元素的值，返回列表
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        # 返回列表，其值为各个类别的IoU
        IoU = intersection / union
        # 求各类别IoU的平均
        mIoU = np.nanmean(IoU)
        return mIoU

    # 同FCN中score.py的fast_hist()函数，用于获取混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # 将gt图像中未对像素进行标记的部分类别移除并进行预测
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

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

# 绘制源图像、生成图像以及目标图像
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# 将元素值从[-1,1]改为[0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# 一列一列地作图
	for i in range(len(images)):
		pyplot.subplot(1, 3, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(images[i])
		pyplot.title(titles[i])
	pyplot.show()

# 加载数据
[X1, X2] = load_real_samples('maps_val_256.npz')
print('Loaded', X1.shape, X2.shape)
# 加载模型
model = load_model('model_010960.h5')
# 随机选择样例
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# 利用源图像生成图像
gen_image = model.predict(src_image)
# 绘制三张图像
plot_images(src_image, gen_image, tar_image)

'''
# 载入图像，化为灰度图并转换成numpy数组形式
im1 = np.array(Image.open("des.jpg").convert('L'))
im2 = np.array(Image.open("out.jpg").convert('L'))
'''
# 将元素值恢复为[0,255]
im1 = gen_image[0] * 127.5 + 127.5
im2 = tar_image[0] * 127.5 + 127.5
im1 = im1.astype(np.uint8)
im2 = im2.astype(np.uint8)
#print(im1.shape)
#print(im2.shape)

# 获取PSNR和SSIM两个指标
psnr = skimage.measure.compare_psnr(im1, im2, 255)
ssim = skimage.measure.compare_ssim(im1, im2, data_range=255, multichannel=True)

print(psnr)
print(ssim)

imgPredict = im1.flatten() # 预测图像，并展平为一维向量
imgLabel = im2.flatten() # 目标图像，并展平为一维向量
metric = SegmentationMetric(256) # 256为像素的不同取值类别数
metric.addBatch(imgPredict, imgLabel)
# 获取语义分割的各个指标
PA = metric.pixelAccuracy()
CPA = metric.classPixelAccuracy()
MPA = metric.meanPixelAccuracy()
MIOU = metric.meanIntersectionOverUnion()

print('PA is : %f' % PA)
#print('CPA is :') # 列表
#print(CPA)
print('MPA is : %f' % MPA)
print('MIOU is : %f' % MIOU)
