# 计算有效接受域的尺寸
def receptive_field(output_size, kernel_size, stride_size):
    return (output_size - 1) * stride_size + kernel_size

print("70x70:")
# 输出层，输出1x1的像素，并且滤波器尺寸为4x4，步幅为1x1
rf = receptive_field(1, 4, 1)
print(rf)
# 倒数第二层，并且滤波器尺寸为4x4，步幅为1x1
rf = receptive_field(rf, 4, 1)
print(rf)
# 3个PatchGAN层，并且滤波器尺寸为4x4，步幅为2x2
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)

print("1x1:")
# 输出层，输出1x1的像素，并且滤波器尺寸为1x1，步幅为1x1
rf = receptive_field(1, 1, 1)
print(rf)
# 倒数第二层，并且滤波器尺寸为1x1，步幅为1x1
rf = receptive_field(rf, 1, 1)
print(rf)
# 1个PatchGAN层，并且滤波器尺寸为1x1，步幅为1x1
rf = receptive_field(rf, 1, 1)
print(rf)

print("16x16:")
# 输出层，输出1x1的像素，并且滤波器尺寸为4x4，步幅为1x1
rf = receptive_field(1, 4, 1)
print(rf)
# 倒数第二层，并且滤波器尺寸为4x4，步幅为1x1
rf = receptive_field(rf, 4, 1)
print(rf)
# 1个PatchGAN层，并且滤波器尺寸为4x4，步幅为2x2
rf = receptive_field(rf, 4, 2)
print(rf)

print("286x286:")
# 输出层，输出1x1的像素，并且滤波器尺寸为4x4，步幅为1x1
rf = receptive_field(1, 4, 1)
print(rf)
# 倒数第二层，并且滤波器尺寸为4x4，步幅为1x1
rf = receptive_field(rf, 4, 1)
print(rf)
# 5个PatchGAN层，并且滤波器尺寸为4x4，步幅为2x2
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)
rf = receptive_field(rf, 4, 2)
print(rf)




