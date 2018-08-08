import random
import os
import numpy as np
from PIL import Image

# 2D filter
def filter_2d(mu,sigma):
    conv_filter_2d = [];
    for i in range(0,5):
         conv_filter_2d.append([])
         for j in range(0,5):
            conv_filter_2d[i].append(np.random.normal(mu,sigma,size=None))
    conv_filter_2d = np.array(conv_filter_2d);
    return conv_filter_2d;

def ReLU(x):
    return abs(x) * (x>0)

def convolution2d(stride,padding,no_of_filters,mu,sigma):
    conv = np.zeros(((im2arr.shape[0] - filter_2d(mu,sigma).shape[0] + 2*padding)/stride + 1, (im2arr.shape[1] - filter_2d(mu,sigma).shape[1] + 2*padding)/stride  + 1, no_of_filters))
    for k in range(0, no_of_filters):
        filter_ = filter_2d(mu,sigma)
        for i in range(0, (im2arr.shape[0] - filter_.shape[0] + 2*padding)/stride  + 1):
            for j in range(0, (im2arr.shape[1] - filter_.shape[1] + 2*padding)/stride  + 1):
                conv[i,j,k] = np.sum(im2arr[i:i+filter_.shape[0],j:j+filter_.shape[1]] * filter_[:,:]) + random.randint(0, 1)
                conv[i,j,k] = ReLU(conv[i,j,k])
    return conv;

def pooling(CONV):
    afterpool = np.zeros((CONV.shape[0]/2, CONV.shape[1]/2, CONV.shape[2]))
    for k in range(0,CONV.shape[2]):
        for i in range(0, CONV.shape[0]/2):
            for j in range(0, CONV.shape[1]/2):
                afterpool[i,j,k] = CONV[2*i,2*j,k]
    return afterpool


def filter_3d(mu,sigma):
    conv_filter_3d = np.zeros((5,5,6))
    for k in range(0,6):
        for i in range(0,5):
            for j in range(0,5):
                conv_filter_3d[i,j,k] = np.random.normal(mu,sigma,size=None);
    return conv_filter_3d


def convolution3d(stride, padding, no_of_filters,mu,sigma):
    conv = np.zeros(((subsample1.shape[0] - filter_3d(mu,sigma).shape[0] + 2*padding)/stride + 1, (subsample1.shape[1] - filter_3d(mu,sigma).shape[1] + 2*padding)/stride  + 1, no_of_filters))
    for n in range(0, no_of_filters):
        filter_ = filter_3d(mu,sigma)
        for k in range(0, (subsample1.shape[2] - filter_.shape[2] + 2*padding)/stride +1):
            for i in range(0, (subsample1.shape[0] - filter_.shape[0] + 2*padding)/stride  + 1):
                for j in range(0, (subsample1.shape[1] - filter_.shape[1] + 2*padding)/stride  + 1):
                    conv[i,j,n] = np.sum(subsample1[i:i+filter_.shape[0],j:j+filter_.shape[1],:] * filter_[:,:,:]) + random.randint(0, 1)
                    conv[i,j,n] = ReLU(conv[i,j,n])
    return conv;

input_imagepath='/home/priyansh/Downloads/sem6/smai/ass2/q1_a/input_img/'
output_imagepath='/home/priyansh/Downloads/sem6/smai/ass2/q1_a/output_img/'
input_images=[]
for img in os.listdir(input_imagepath):
    input_images.append(img)
for img in input_images:
    newpath=output_imagepath+img
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    myImage = Image.open(input_imagepath+img).convert('L');
    print myImage;
    im2arr = np.array(myImage);
    print im2arr.shape;
    myImage.show();

    mu1 = 0;
    sigma1 = 10;
    CONV1 = convolution2d(1,0,6,mu1,sigma1);
    subsample1 = pooling(CONV1);
    num=0;
    for i in range(6):
         img = Image.fromarray(subsample1[:,:,i],'RGB')
         img.save(newpath+"/"+'1_'+str(num)+'.jpg')
         num+=1


    mu2 = 0;
    sigma2 = 10;
    CONV2 = convolution3d(1, 0, 16,mu2,sigma2);
    subsample2 = pooling(CONV2);

    num=0;
    for j in range(16):
         img = Image.fromarray(subsample2[:,:,j],'RGB')
         img.save(newpath+"/"+'2_'+str(num)+'.jpg')
         num+=1

