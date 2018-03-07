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
from ctm_map import *
import pickle
global DEBUG
DEBUG = False

def processImageCrop(im_info, transformer, deep, ctm, map_ctm, extractor):
    
    im_path = im_info[0]
    ctm_path = im_info[1]
    #print ctm_path
    im_reshape = im_info[2]
    ctm_des = []
    processed_image = np.zeros((1,1))
    map_ctm_fea = np.zeros((1,1))
    if ctm == True:
        ctm_des = np.load(ctm_path)
    if deep == True or map_ctm['flag'] == True:
        data_in = caffe.io.load_image(im_path)
        if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
            data_in = caffe.io.resize_image(data_in, im_reshape)
    if deep == True:
        processed_image = transformer.preprocess('data_in', data_in)
        # print processed_image.dtype
    if map_ctm['flag'] == True:
        map_ctm_path = im_path.replace(map_ctm['data_dir'],map_ctm['ctm_dir'])[:-4]+'.npy'
        if os.path.isfile(map_ctm_path):
            map_ctm_fea = np.load(map_ctm_path)
            # print 'load..'
        else:
            map_ctm_folder = map_ctm_path.replace(map_ctm_path.split('/')[-1],'')
            if not os.path.exists(map_ctm_folder):
                os.makedirs(map_ctm_folder)
            map_ctm_fea = extractor.extract_feature(im_path)
            np.save(map_ctm_path, map_ctm_fea)
        
    res_dic = {}
    # print len(ctm_des)
    res_dic['ctm']= ctm_des
    res_dic['map_ctm']= map_ctm_fea
    
    if DEBUG:
        print map_ctm_fea[0,:,:]
        map_lom = cv2.resize(map_ctm_fea[0,:, :], (80, 160), interpolation=cv2.INTER_LINEAR)
        map_lom = map_lom.astype('uint8')
        cv2.imshow('ctm',map_lom)
        cv2.waitKey(1000)
    res_dic['pro_img'] = processed_image
    return res_dic


class ImageProcessorCrop(object):
    def __init__(self, transformer, deep, ctm, map_ctm, extractor):
        self.transformer = transformer
        self.deep = deep
        self.ctm = ctm
        self.map_ctm = map_ctm
        self.extractor = extractor

    def __call__(self, im_info):
        return processImageCrop(im_info, self.transformer, self.deep, self.ctm, self.map_ctm, self.extractor)


class sequenceGeneratorVideo(object):
	def __init__(self, buffer_size, num_videos, video_cache_dict, video_order):
	    self.buffer_size = buffer_size
	    self.N = self.buffer_size
	    self.num_videos = num_videos
	    self.video_cache_dict = video_cache_dict
	    self.video_order = video_order
	    self.idx = 0
	    self.point = self.num_videos/100
	def __call__(self):
		label_r = []
		im_paths = []
		im_paths_p = []
		ctm_paths = []
		ctm_paths_p = []
		im_reshape = []
		idx_list = []
		load_cache_key = ''
		idx_list = range(self.idx, self.idx + self.buffer_size)
		# print idx_list
		if self.idx + self.buffer_size >= self.num_videos:
			idx_list = range(self.idx, self.num_videos)
			new_end = self.buffer_size - (self.num_videos - self.idx)
			new_list = range(0, new_end)
			idx_list.extend(new_list)
			self.idx = idx_list[-1]+1-self.buffer_size 
		# print idx_list
		for i in idx_list:
			now_cache_key = '%d'%(i/self.point)
			if not now_cache_key == load_cache_key:
				load_cache_key = now_cache_key
				video_dict = self.readPKL(self.video_cache_dict[load_cache_key])
			# print [i,self.point,load_cache_key]
			key = self.video_order[i]
			label = video_dict[key]['label']
			video_reshape = video_dict[key]['reshape']
			label_r.extend([label])

			im_reshape.extend([(video_reshape)])
			frames = []
			frames.extend(video_dict[key]['frames'])
			im_paths.extend(frames)
			ctms = []
			ctms.extend(video_dict[key]['ctm'])
			ctm_paths.extend(ctms)

			frames_p = []
			frames_p.extend(video_dict[key]['frames_p'])
			im_paths_p.extend(frames_p)

			ctms_p = []
			ctms_p.extend(video_dict[key]['ctm_p'])
			ctm_paths_p.extend(ctms_p)

		im_info = zip(im_paths, ctm_paths, im_reshape)
		im_info_p = zip(im_paths_p, ctm_paths_p, im_reshape)

		self.idx += self.buffer_size
		# print self.idx
		if self.idx >= self.num_videos:
		    self.idx = self.idx - self.buffer_size

		return label_r, im_info , im_info_p
	def readPKL(self, PKL_name):
		# print PKL_name
		pkl_file = open(PKL_name,'rb')
		video_dict = pickle.load(pkl_file)
		pkl_file.close()
		return video_dict


def advance_batch(result, sequence_generator, image_processor, pool):
    label_r, im_info ,im_info_p = sequence_generator()
    #print im_info
    tmp = image_processor(im_info[0])
    
    buffer_result = pool.map(image_processor, im_info)
    result['data'] = []
    result['ctm'] = []
    result['map_ctm'] = []

    for buf_ in buffer_result:
        result['data'].append(buf_['pro_img'])
        result['ctm'].append(buf_['ctm'])
        result['map_ctm'].append(buf_['map_ctm'])
    #print len(result['data'])
    buffer_result1 = pool.map(image_processor, im_info_p)
    result['data_p'] = []
    result['ctm_p'] = []
    result['map_ctm_p'] = []
    for buf_ in buffer_result1:
        result['data_p'].append(buf_['pro_img'])
        result['ctm_p'].append(buf_['ctm'])
        result['map_ctm_p'].append(buf_['map_ctm'])
    
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
        # self.video_dict = {}
        self.video_cache_dict = {}
        self.video_order = []
        self.num_videos = len(self.video_order)
        self.ctm = False # preproc the data in affine style
        self.ctm_dim = 26960 
        self.multylabel = False
        self.deep = True
        self.map_ctm = {}

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
        
        self.extractor = feature_extractor(RGB_para = self.map_ctm['RGB_para'],HSV_para=self.map_ctm['HSV_para'],SILTP_para=self.map_ctm['SILTP_para'],\
                                          block_size = self.map_ctm['block_size'], block_step = self.map_ctm['block_step'], pad_size = self.map_ctm['pad_size'], \
                                          tau = self.map_ctm['tau'], R = self.map_ctm['R'], numPoints = self.map_ctm['numPoints'])

        self.thread_result = {}
        self.thread = None
        pool_size = 24
        
        
        self.image_processor = ImageProcessorCrop(self.transformer, self.deep, self.ctm, self.map_ctm, self.extractor)
        self.sequence_generator = sequenceGeneratorVideo(self.buffer_size, self.num_videos, self.video_cache_dict, self.video_order)

        self.pool = Pool(processes=pool_size, maxtasksperchild = 40)
        self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, \
                                            self.pool)
        self.dispatch_worker()
        self.top_names = []
        if self.deep == True:
            self.top_names.extend(['data', 'data_p']) 
        if self.ctm == True:
            self.top_names.extend(['data_ctm','data_ctm_p'])
        if self.map_ctm['flag'] == True:
            self.top_names.extend(['data_map_ctm','data_map_ctm_p'])
            map_width = (self.width+2*self.map_ctm['pad_size'])/self.map_ctm['block_step']-1
            map_height = (self.height+2*self.map_ctm['pad_size'])/self.map_ctm['block_step']-1
            map_channels = 0
            if self.map_ctm['RGB_para'][0]:
                map_channels += self.map_ctm['RGB_para'][1]*self.channels
            if self.map_ctm['HSV_para'][0]:
                map_channels += self.map_ctm['HSV_para'][1]*self.channels
            if self.map_ctm['SILTP_para'][0]:
                map_channels += self.map_ctm['SILTP_para'][1]

            
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
            elif name == 'data_ctm' or name == 'data_ctm_p':
                shape = (self.N, self.ctm_dim)
            elif name == 'data_map_ctm' or name == 'data_map_ctm_p':
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
        new_result_ctm = self.thread_result['ctm']
        new_result_map_ctm = self.thread_result['map_ctm']
        new_result_data_p = self.thread_result['data_p']
        new_result_ctm_p = self.thread_result['ctm_p']
        new_result_map_ctm_p = self.thread_result['map_ctm_p']
        new_result_label = self.thread_result['label']
        # for ii in range(self.buffer_size):
        #     old_idx = ii
        #     new_idx = ii
        #     new_result_data[new_idx] = self.thread_result['data'][old_idx]
        #     new_result_data_p[new_idx] = self.thread_result['data_p'][old_idx]
        #     new_result_ctm[new_idx] = self.thread_result['ctm'][old_idx]
        #     new_result_ctm_p[new_idx] = self.thread_result['ctm_p'][old_idx]
        #     new_result_label[new_idx] = self.thread_result['label'][old_idx]
        label_array = np.array(new_result_label)
        for top_index, name in zip(range(len(top)), self.top_names):
            if name == 'data':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_data[i]
            elif name == 'data_ctm':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_ctm[i]
            elif name == 'data_map_ctm':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_map_ctm[i]
            elif name == 'data_p':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_data_p[i]
            elif name == 'data_ctm_p':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_ctm_p[i]
            elif name == 'data_map_ctm_p':
                for i in range(self.N):
                    top[top_index].data[i, ...] = new_result_map_ctm_p[i]
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
