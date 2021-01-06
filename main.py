# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from keras import backend as K
import tensorflow as tf
import numpy as np
import cv2

# load the model
# vgg_model = VGG16()
# output_layer_list = [2, 9, 17]
# outputs = [vgg_model.layers[idx].output for idx in output_layer_list]
# model = Model(inputs=vgg_model.inputs, outputs=outputs)
# # # load the image with the required shape
#
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
#
# net_input_size = (224,224)
# frame = load_img('cow.jpg', target_size=net_input_size)
# frame = img_to_array(frame)
# frame = expand_dims(frame, axis=0)
# frame = preprocess_input(frame)
# feature_maps = model.predict(frame)
#
# import math
# item_per_col = int(math.sqrt(feature_maps[0].shape[3])) # Căn bậc 2 của 64 là 8
# for fm in feature_maps:
# 	idx = 1
# 	for _ in range(item_per_col):
# 		for _ in range(item_per_col):
# 			ax = pyplot.subplot(item_per_col, item_per_col, idx)
# 			ax.set_xticks([])
# 			ax.set_yticks([])
# 			pyplot.imshow(fm[0, :, :, idx-1])
# 			idx += 1
# 	pyplot.show()
# #
model = VGG16()
model.summary()
file_name = 'girl1.jpg'
# load ảnh
net_input_size = (224,224)
frame = load_img(file_name, target_size=net_input_size)
frame = img_to_array(frame)
frame = expand_dims(frame, axis=0)
frame = preprocess_input(frame)

with tf.GradientTape() as tape:
	# Tạo ra một model mới có 1 input và 2 output là output của model và output của conv layer cuối cùng
	last_conv_layer = model.get_layer('block5_conv3')
	new_model = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])

	# Đưa ảnh vào model mới để lấy output
	model_out, last_conv_layer = new_model(frame)

	# Lấy output có prob lớn nhất
	class_out = model_out[:, np.argmax(model_out[0])]

	# Tính gradient của class output đối với output của last_conv_layer
	grads = tape.gradient(class_out, last_conv_layer)


	# Tính giá trị trung bình của gradient, kết quả là 1 vector 512
	pooled_grads = K.mean(grads, axis=(0, 1, 2))


# Nhân pooled_grads với output của last_conv_layer và lấy mean để có heatmap.
# Chú ý last_conv_layer có size (1, 14,14,512)
# Output là heatmap size (1,14,14)
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

# Xử lý heat map, bỏ giá trị âm, scale lại giá trị về 0,1
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((14, 14))

# Vẽ heatmap lên ảnh
# Đọc ảnh con bò
img = cv2.imread(file_name)

# Chỉnh lại heatmap
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

# Vẽ heatmap lên ảnh
overlay_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
cv2.imshow("B",overlay_img)
cv2.waitKey()