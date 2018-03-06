#!/usr/bin/env python

# Data layer for video.  Change flow_frames and RGB_frames to be the path to the flow and RGB frames.

import sys

sys.path.append('caffe/python') #should verify
import cv2
import caffe
import numpy as np

import random
import os
from multiprocessing import Pool
from threading import Thread
from lomo_map import *
global DEBUG
DEBUG = False
# def channel_hist(img_channel, bin_size):
#     hist = cv2.calcHist([img_channel],[0],None,[bin_size],[0.0,255.0])
#     w,h = img_channel.shape
#     return hist/(w*h)

# def map_lomo_com(im,block_step,block_size,bin_size):
#     h,w,c = im.shape
#     map_height = h/block_step-1
#     map_width = w/block_step-1
#     #print [map_height,map_width]
#     new_array = np.zeros((map_height,map_width,bin_size*c),dtype='float32')
#     for i in range(map_height):
#         for j in range(map_width):
#             buff = im[i*block_step:i*block_step+block_size,j*block_step:j*block_step+block_size,:]
#             for k in range(c):
#                 hist_buff = channel_hist(buff[:,:,k], bin_size)
#                 new_array[i,j,k*bin_size:(k+1)*bin_size] = hist_buff[:,0]
#     return new_array.transpose(2,0,1)*255

# def format_time(num_time):
#     seconds=num_time%60
#     minuites=num_time/60%60
#     hours=num_time/3600%24
#     days=num_time/3600/24
#     fm_time={'sec':seconds,'min':minuites,'hours':hours,'days':days}
#     return fm_time


def processImageCrop(im_info, transformer, deep, lomo, map_lomo):
    im_path = im_info[0]
    
    lomo_path = im_info[1]
    #print lomo_path
    im_reshape = im_info[2]
    lomo_des = []
    processed_image = np.zeros((1,1))
    map_lomo_fea = np.zeros((1,1))
    if lomo == True:
        lomo_des = np.load(lomo_path)
    if deep == True or map_lomo['flag'] == True:
        data_in = caffe.io.load_image(im_path)
        if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
            data_in = caffe.io.resize_image(data_in, im_reshape)
    if deep == True:
        processed_image = transformer.preprocess('data_in', data_in)
        # print processed_image.dtype
    if map_lomo['flag'] == True:
        map_lomo_path = im_path.replace('data_aug','lomo_map')[:-4]+'.npy'
        if os.path.isfile(map_lomo_path):
            map_lomo_fea = np.load(map_lomo_path)
            # print 'load..'
        else:
            pad_size = map_lomo['pad_size']
            data_in = cv2.imread(im_path,cv2.CV_LOAD_IMAGE_COLOR)
            data_in = cv2.resize(data_in,(80,160),interpolation=cv2.INTER_LINEAR)
            im = np.pad(data_in,((pad_size,pad_size),(pad_size,pad_size),(0,0)),mode='constant',constant_values=0)
            map_lomo_fea = map_lomo_com(im,map_lomo['block_step'],map_lomo['block_size'],map_lomo['bin_size'])
            np.save(map_lomo_path, map_lomo_fea)
        
    res_dic = {}
    # print len(lomo_des)
    res_dic['lomo']= lomo_des
    res_dic['map_lomo']= map_lomo_fea
    
    if DEBUG:
        print map_lomo_fea[0,:,:]
        map_lom = cv2.resize(map_lomo_fea[0,:, :], (80, 160), interpolation=cv2.INTER_LINEAR)
        map_lom = map_lom.astype('uint8')
        cv2.imshow('lomo',map_lom)
        cv2.waitKey(1000)
    res_dic['pro_img'] = processed_image
    return res_dic


class ImageProcessorCrop(object):
    def __init__(self, transformer, deep, lomo, map_lomo):
        self.transformer = transformer
        self.deep = deep
        self.lomo = lomo
        self.map_lomo = map_lomo

    def __call__(self, im_info):
        return processImageCrop(im_info, self.transformer, self.deep, self.lomo, self.map_lomo)


class sequenceGeneratorVideo(object):
    def __init__(self, buffer_size, num_videos, video_dict, video_order):
        self.buffer_size = buffer_size
        self.N = self.buffer_size
        self.num_videos = num_videos
        self.video_dict = video_dict
        self.video_order = video_order
        self.idx = 0

    def __call__(self):
        label_r = []
        im_paths = []
        im_paths_p = []
        lomo_paths = []
        lomo_paths_p = []
        im_reshape = []

        if self.idx + self.buffer_size >= self.num_videos:
            idx_list = range(self.idx, self.num_videos)
            idx_list.extend(range(0, self.buffer_size - (self.num_videos - self.idx)))
        else:
            idx_list = range(self.idx, self.idx + self.buffer_size)

        for i in idx_list:
            key = self.video_order[i]
            label = self.video_dict[key]['label']
            video_reshape = self.video_dict[key]['reshape']
            label_r.extend([label])

            im_reshape.extend([(video_reshape)])
            frames = []
            frames.extend(self.video_dict[key]['frames'])
            im_paths.extend(frames)
            lomos = []
            lomos.extend(self.video_dict[key]['lomo'])
            lomo_paths.extend(lomos)

            frames_p = []
            frames_p.extend(self.video_dict[key]['frames_p'])
            im_paths_p.extend(frames_p)

            lomos_p = []
            lomos_p.extend(self.video_dict[key]['lomo_p'])
            lomo_paths_p.extend(lomos_p)

        im_info = zip(im_paths, lomo_paths, im_reshape)
        im_info_p = zip(im_paths_p, lomo_paths_p, im_reshape)

        self.idx += self.buffer_size
        if self.idx >= self.num_videos:
            self.idx = self.idx - self.num_videos

        return label_r, im_info , im_info_p


def advance_batch(result, sequence_generator, image_processor, pool):
    label_r, im_info ,im_info_p = sequence_generator()
    #print im_info
    tmp = image_processor(im_info[0])
    
    buffer_result = pool.map(image_processor, im_info)
    result['data'] = []
    result['lomo'] = []
    result['map_lomo'] = []

    for buf_ in buffer_result:
        result['data'].append(buf_['pro_img'])
        result['lomo'].append(buf_['lomo'])
        result['map_lomo'].append(buf_['map_lomo'])
    #print len(result['data'])
    buffer_result1 = pool.map(image_processor, im_info_p)
    result['data_p'] = []
    result['lomo_p'] = []
    result['map_lomo_p'] = []
    for buf_ in buffer_result1:
        result['data_p'].append(buf_['pro_img'])
        result['lomo_p'].append(buf_['lomo'])
        result['map_lomo_p'].append(buf_['map_lomo'])
    
    result['label'] = label_r


class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
        self.result = result
        self.sequence_generator = sequence_generator
        self.image_processor = image_processor
        self.pool = pool
    def __call__(self):
        return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool)


class videoRead(caffe.Layer):
    def initialize(self):
        self.train_or_test = 'train'
        self.buffer_size = 12  # num videos processed per batch
        self.N = self.buffer_size
        self.idx = 0
        self.channels = 3
        self.height = 227
        self.width = 227
        self.path_to_images = './' # the pre path to the datasets
        self.video_dict = {}
        self.video_order = []
        self.num_videos = len(self.video_dict)
        self.lomo = False # preproc the data in affine style
        self.lomo_dim = 26960 
        self.multylabel = False
        self.deep = True
        self.map_lomo = {}

    def setup(self, bottom, top):
        random.seed(10)
        self.initialize()

        # set up data transformer
        shape = (self.N, self.channels, self.height, self.width)

        self.transformer = caffe.io.Transformer({'data_in': shape})
        self.transformer.set_raw_scale('data_in', 255)
        image_mean = [104, 117, 123]
        # self.transformer.set_is_flow('data_in', False)
        channel_mean = np.zeros((3, self.height, self.width))
        for channel_index, mean_val in enumerate(image_mean):
            channel_mean[channel_index, ...] = mean_val
        self.transformer.set_mean('data_in', channel_mean)
        self.transformer.set_channel_swap('data_in', (2, 1, 0))
        self.transformer.set_transpose('data_in', (2, 0, 1))

        self.thread_result = {}
        self.thread = None
        pool_size = 1

        self.image_processor = ImageProcessorCrop(self.transformer, self.deep, self.lomo, self.map_lomo)
        self.sequence_generator = sequenceGeneratorVideo(self.buffer_size, self.num_videos,
                                                         self.video_dict, self.video_order)

        self.pool = Pool(processes=pool_size)
        self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, \
                                            self.pool)
        self.dispatch_worker()
        self.top_names = []
        if self.deep == True:
            self.top_names.extend(['data', 'data_p']) 
        if self.lomo == True:
            self.top_names.extend(['data_lomo','data_lomo_p'])
        if self.map_lomo['flag'] == True:
            self.top_names.extend(['data_map_lomo','data_map_lomo_p'])
            map_width = (self.width+2*self.map_lomo['pad_size'])/self.map_lomo['block_step']-1
            map_height = (self.height+2*self.map_lomo['pad_size'])/self.map_lomo['block_step']-1
            map_channels = self.map_lomo['bin_size']*self.channels
        self.top_names.extend(['label'])

        if self.multylabel == True:
            self.top_names.extend(['label_ID','label_pID'])
        
        print 'Outputs:', self.top_names
        if len(top) != len(self.top_names):
            raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                            (len(self.top_names), len(top)))
        self.join_worker()
        for top_index, name in enumerate(self.top_names):
            if name == 'data' or name == 'data_p':
                shape = (self.N, self.channels, self.height, self.width)
            elif name == 'label' or name == 'label_ID' or name == 'label_pID':
                shape = (self.N,)
            elif name == 'data_lomo' or name == 'data_lomo_p':
                shape = (self.N, self.lomo_dim)
            elif name == 'data_map_lomo' or name == 'data_map_lomo_p':
                shape = (self.N, map_channels, map_height, map_width)
            top[top_index].reshape(*shape)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):

        if self.thread is not None:
            self.join_worker()
        # print self.thread_result.keys()
        # rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
        new_result_data = self.thread_result['data']
        new_result_lomo = self.thread_result['lomo']
        new_result_map_lomo = self.thread_result['map_lomo']
        new_result_data_p = self.thread_result['data_p']
        new_result_lomo_p = self.thread_result['lomo_p']
        new_result_map_lomo_p = self.thread_result['map_lomo_p']
        new_result_label = self.thread_result['label']
        # for ii in range(self.buffer_size):
        #     old_idx = ii
        #     new_idx = ii
        #     new_result_data[new_idx] = self.thread_result['data'][old_idx]
        #     new_result_data_p[new_idx] = self.thread_result['data_p'][old_idx]
        #     new_result_lomo[new_idx] = self.thread_result['lomo'][old_idx]
        #     new_result_lomo_p[new_idx] = self.thread_result['lomo_p'][old_idx]
        #     new_result_label[new_idx] = self.thread_result['label'][old_idx]
        label_array = np.array(new_result_label)
        for top_index, name in zip(range(len(top)), self.top_names):
            if name == 'data':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_data[i]
            elif name == 'data_lomo':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_lomo[i]
            elif name == 'data_map_lomo':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_map_lomo[i]
            elif name == 'data_p':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_data_p[i]
            elif name == 'data_lomo_p':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_lomo_p[i]
            elif name == 'data_map_lomo_p':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_map_lomo_p[i]
            elif name == 'label' :
                top[top_index].data[...] = label_array[:,0]
                #print label_array[:,0]
            elif name == 'label_ID':
                top[top_index].data[...] = label_array[:,1]
                #print label_array[:,1]
            elif name == 'label_pID':
                #print label_array[:,2]
                top[top_index].data[...] = label_array[:,2]
        #print top[2].data[...] 
        self.dispatch_worker()

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        pass