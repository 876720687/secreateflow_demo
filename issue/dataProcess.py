import os,glob
import numpy as np
import cv2

# root='flowers'
root='flower_photos/'
classes=['daisy','dandelion','rose','sunflower','tulip']
img_paths=[]#保存所有图片路径
labels=[]#保存图片的类别标签(0,1,2,3,4)
for i,cls in enumerate(classes):
    cls_img_paths=glob.glob(os.path.join(root,cls,"*.jpg"))
    img_paths.extend(cls_img_paths)
    labels.extend([i]*len(cls_img_paths))

#图片->numpy
img_numpys=[]
labels=np.array(labels)
for img_path in img_paths:
    img_numpy = cv2.imread(img_path)
    img_numpy = cv2.resize(img_numpy, (240, 240))
    img_numpy=np.reshape(img_numpy,(1,240,240,3))
    img_numpys.append(img_numpy)

images=np.concatenate(img_numpys,axis=0)
print(images.shape)#(4317, 240, 240, 3)
print(labels.shape)#(4317,)

#打乱顺序
state = np.random.get_state()
np.random.shuffle(images)
np.random.set_state(state)
np.random.shuffle(labels)

print(images.shape)#(4317, 240, 240, 3)
print(labels.shape)#(4317,)
# 方法1
# 直接将数据进行训练集和测试集的拆分，并不分配结点
per = 0.7
x_train=images[:int(per*images.shape[0]),:,:,:]
x_test=images[int(per*images.shape[0]):,:,:,:]
y_train=labels[:int(per*labels.shape[0])]
y_test=labels[int(per*labels.shape[0]):]
np.save("x_train1.npy",x_train)
np.save("x_test1.npy",x_test)
np.save("y_train1.npy",y_train)
np.save("y_test1.npy",y_test)




# 方法2
#给两个节点分配images和labels，各分配50%的数据
# per=0.5
# node1_images=images[:int(per*images.shape[0]),:,:,:]
# node1_label=labels[:int(per*images.shape[0])]
# node2_images=images[int(per*images.shape[0]):,:,:,:]
# node2_label=labels[int(per*images.shape[0]):]
# print(node1_images.shape,node1_label.shape,node2_images.shape,node2_label.shape)#(2158, 240, 240, 3) (2158,) (2159, 240, 240, 3) (2159,)
#
# #每个节点拆分训练集和验证集
# per=0.7
# x_train1=node1_images[:int(per*node1_images.shape[0]),:,:,:]
# x_test1=node1_images[int(per*node1_images.shape[0]):,:,:,:]
# y_train1=node1_label[:int(per*node1_label.shape[0])]
# y_test1=node1_label[int(per*node1_label.shape[0]):]
# print(x_train1.shape,x_test1.shape,y_train1.shape,y_test1.shape)#(1510, 240, 240, 3) (648, 240, 240, 3) (1510,) (648,)
#
# x_train2=node2_images[:int(per*node2_images.shape[0]),:,:,:]
# x_test2=node2_images[int(per*node2_images.shape[0]):,:,:,:]
# y_train2=node2_label[:int(per*node2_label.shape[0])]
# y_test2=node2_label[int(per*node2_label.shape[0]):]
# print(x_train2.shape,x_test2.shape,y_train2.shape,y_test2.shape)#(1511, 240, 240, 3) (648, 240, 240, 3) (1511,) (648,)
#
# #保存为npy文件
# np.save("x_train1.npy",x_train1)
# np.save("x_test1.npy",x_test1)
# np.save("y_train1.npy",y_train1)
# np.save("y_test1.npy",y_test1)
# np.save("x_train2.npy",x_train2)
# np.save("x_test2.npy",x_test2)
# np.save("y_train2.npy",y_train2)
# np.save("y_test2.npy",y_test2)
#
