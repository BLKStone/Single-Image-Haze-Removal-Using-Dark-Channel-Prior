# -*- coding: utf-8 -*-
' a module for a dark channel based algorithm which remove haze on picture '

__author__ = 'Ray'

import math
import numpy as np
import cv2

# 用于排序时存储原来像素点位置的数据结构
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value

	def printInfo(self):
		print '%s:%s:%s' %(self.x,self.y,self.value)

# 获取最小值矩阵
# 获取BGR三个通道的最小值
def getMinChannel(img):

	# 输入检查
	if len(img.shape)==3 and img.shape[2]==3:
		pass
	else:
		print "bad image shape, input must be color image"
		return None

	imgGray = np.zeros((img.shape[0],img.shape[1]),dtype = np.uint8)
	localMin = 255

	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
			localMin = 255
			for k in range(0,3):
				if img.item((i,j,k)) < localMin:
					localMin = img.item((i,j,k))
			imgGray[i,j] = localMin

	return imgGray

# 获取暗通道
def getDarkChannel(img,blockSize = 3):

	# 输入检查
	if len(img.shape)==2:
		pass
	else:
		print "bad image shape, input image must be two demensions"
		return None

	# blockSize检查
	if blockSize % 2 == 0 or blockSize < 3:
		print 'blockSize is not odd or too small'
		return None

	# 计算addSize
	addSize = (blockSize-1)/2

	newHeight = img.shape[0] + blockSize - 1
	newWidth = img.shape[1] + blockSize - 1

	# 中间结果
	imgMiddle = np.zeros((newHeight,newWidth))
	imgMiddle[:,:] = 255

	imgMiddle[addSize:newHeight - addSize,addSize:newWidth - addSize] = img

	imgDark = np.zeros((img.shape[0],img.shape[1]),np.uint8)
	localMin = 255

	for i in range(addSize,newHeight - addSize):
		for j in range(addSize,newWidth - addSize):
			localMin = 255
			for k in range(i-addSize,i+addSize+1):
				for l in range(j-addSize,j+addSize+1):
					if imgMiddle.item((k,l)) < localMin:
						localMin = imgMiddle.item((k,l))
			imgDark[i-addSize,j-addSize] = localMin

	return imgDark

# 获取全局大气光强度
def getAtomsphericLight(darkChannel,img,meanMode = False, percent = 0.001):

	size = darkChannel.shape[0]*darkChannel.shape[1]
	height = darkChannel.shape[0]
	width = darkChannel.shape[1]

	nodes = []

	# 用一个链表结构(list)存储数据
	for i in range(0,height):
		for j in range(0,width):
			oneNode = Node(i,j,darkChannel[i,j])
			nodes.append(oneNode)	

	# 排序
	nodes = sorted(nodes, key = lambda node: node.value,reverse = True)
	
	atomsphericLight = 0

	# 原图像像素过少时，只考虑第一个像素点
	if int(percent*size) == 0:
		for i in range(0,3):
			if img[nodes[0].x,nodes[0].y,i] > atomsphericLight:
				atomsphericLight = img[nodes[0].x,nodes[0].y,i]

		return atomsphericLight

	# 开启均值模式
	if meanMode:
		sum = 0
		for i in range(0,int(percent*size)):
			for j in range(0,3):
				sum = sum + img[nodes[i].x,nodes[i].y,j]

		
		atomsphericLight = int(sum/(int(percent*size)*3))
		return atomsphericLight

	# 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值
	for i in range(0,int(percent*size)):
		for j in range(0,3):
			if img[nodes[i].x,nodes[i].y,j] > atomsphericLight:
				atomsphericLight = img[nodes[i].x,nodes[i].y,j]

	return atomsphericLight

# 恢复原图像
# Omega 去雾比例 参数
# t0 最小透射率值
def getRecoverScene(img,omega = 0.95,t0 = 0.1 , blockSize = 15 , meanMode = False,percent = 0.001):

	imgGray = getMinChannel(img)
	imgDark = getDarkChannel(imgGray, blockSize = blockSize)
	atomsphericLight = getAtomsphericLight(imgDark,img,meanMode = meanMode,percent= percent)

	imgDark = np.float64(imgDark)
	transmission = 1 - omega * imgDark / atomsphericLight

	# 防止出现t小于0的情况
	# 对t限制最小值为0.1
	for i in range(0,transmission.shape[0]):
		for j in range(0,transmission.shape[1]):
			if transmission[i,j] < 0.1:
				transmission[i,j] = 0.1

	sceneRadiance = np.zeros(img.shape)

	for i in range(0,3):
		img = np.float64(img)
		sceneRadiance[:,:,i] = (img[:,:,i] - atomsphericLight)/transmission + atomsphericLight

		# 限制透射率 在0～255
		for j in range(0,sceneRadiance.shape[0]):
			for k in range(0,sceneRadiance.shape[1]):
				if sceneRadiance[j,k,i] > 255:
					sceneRadiance[j,k,i] = 255
				if sceneRadiance[j,k,i] < 0:
					sceneRadiance[j,k,i]= 0

	sceneRadiance = np.uint8(sceneRadiance)

	return sceneRadiance

# 调用示例

def sample():
	img = cv2.imread('tiananmen1.bmp',cv2.IMREAD_COLOR)
	sceneRadiance = getRecoverScene(img)

	cv2.imshow('original',img)
	cv2.imshow('test',sceneRadiance)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

sample()