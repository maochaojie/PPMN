from math import *
import glob
import random
import sys
import os
#import caffe

from multiprocessing import Pool 
sys.path.append('./experiments_py_lomo/common_tools')
from base_input_layer import videoRead
from ProgressBar import *

def processList(filelistname,height,width):
    video_dict = {}
    videokey = int(filelistname.split(' ')[0])
    video_dict[videokey] = {}
    video_dict[videokey]['frames'] = []
    video_dict[videokey]['frames_p'] = []
    video_dict[videokey]['lomo'] = []
    video_dict[videokey]['lomo_p'] = []

    filename = filelistname.split(' ')[1]
    person_id = int(filename[-13:-9])
    #print person_id
    # print fileTem
    video_dict[videokey]['frames'].append(filelistname.split(' ')[1])

    filename = filelistname.split(' ')[2]
    person_pid = int(filename[-13:-9])
    
    video_dict[videokey]['frames_p'].append(filelistname.split(' ')[2])  # should verify
    
    video_dict[videokey]['lomo'].append((filelistname.split(' ')[1][:-4]+'.npy').replace('data_aug','lomo_feature'))
    video_dict[videokey]['lomo_p'].append((filelistname.split(' ')[2][:-4]+'.npy').replace('data_aug','lomo_feature'))

    video_dict[videokey]['reshape'] = (height, width)
    video_dict[videokey]['label'] = []
    video_dict[videokey]['label'].append(int(filelistname.split(' ')[3]))
    video_dict[videokey]['label'].append(person_id)
    video_dict[videokey]['label'].append(person_pid)

    return video_dict


class listprocessor(object):
    def __init__(self,height,width):
        self.height = height
        self.width = width

    def __call__(self, filename):
        return processList(filename, self.height, self.width)

def readlistFromFile(video_list,height,width):

    f = open(video_list, 'r')
    f_lines = f.readlines()
    # print f_lines
    f.close()
    # f = open(num_list, 'r')
    # f_list = f.readlines()
    # # print f_lines
    # f.close()

    # num_dict = {}
    # for line in f_list:
    #     num_dict[line.split(' ')[0]] = int(line.split(' ')[1])
    pool_size = 24
    pool = Pool(processes=pool_size)
    processor = listprocessor(height,width)
    
    video_dict = {}
    video_dicList = []
    new_lines = []
    video_order = []
    c = 0
    total = len(f_lines)
    point = total/100
    for ix, line in enumerate(f_lines):
        newline = str(ix)+' '+line 
        new_lines.append(newline)
        if ix%point == 0 and ix > 0:
            # print len(new_lines)
            video_dict_buffer = pool.map(processor,new_lines)
            video_dicList.extend(video_dict_buffer)
            
            new_lines = []
            progresslog('data list is loading',ix/point)
    video_dict_buffer = pool.map(processor,new_lines)
    video_dicList.extend(video_dict_buffer)
    # video_order.extend(video_order)
    pool.close()
    pool.join()
    for entity in video_dicList:
        key = entity.keys()[0] 
        video_dict[key] = entity[entity.keys()[0]]
    # video_dict = video_dicList
    video_order = video_dict.keys()
    # print video_order
    print 'video_list:%s'%(video_list)
    print 'list example:'
    print random.choice(video_dict)
    return [video_dict,video_order]

def readlistFromFileToCache(video_list,height,width):
    import pickle
    import time
    f = open(video_list, 'r')
    f_lines = f.readlines()
    # print f_lines
    f.close()
    # f = open(num_list, 'r')
    # f_list = f.readlines()
    # # print f_lines
    # f.close()

    # num_dict = {}
    # for line in f_list:
    #     num_dict[line.split(' ')[0]] = int(line.split(' ')[1])
    pool_size = 24
    pool = Pool(processes=pool_size)
    processor = listprocessor(height,width)
    
    video_dict = {}
    video_dicList = []
    new_lines = []
    video_order = []
    c = 0
    total = len(f_lines)
    point = total/100
    for ix, line in enumerate(f_lines):
        newline = str(ix)+' '+line 
        new_lines.append(newline)
        if ix%point == 0 and ix > 0:
            # print len(new_lines)
            video_dict_buffer = pool.map(processor,new_lines)
            video_dicList.extend(video_dict_buffer)
            
            new_lines = []
            progresslog('data list is loading',ix/point)
    video_dict_buffer = pool.map(processor,new_lines)
    video_dicList.extend(video_dict_buffer)
    # video_order.extend(video_order)
    pool.close()
    pool.join()
    #write to cache
    time_stamp = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    cache_folder = '%s/%s'%('cache',time_stamp)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    ix = 0
    video_cache_dict = {}
    for entity in video_dicList:
        key = entity.keys()[0] 
        video_dict[key] = entity[entity.keys()[0]]
        ix+=1
        if ix%point == 0 and ix >0:
            #write to cache
            cache_name = '%s/%d.pkl'%(cache_folder,ix/point-1)
            cache_key = '%d'%(ix/point-1)
            
            out_pkl = open(cache_name, 'wb')
            pickle.dump(video_dict, out_pkl)
            out_pkl.close()

            video_cache_dict[cache_key] = cache_name
            video_order_buffer = video_dict.keys()
            video_order.extend(video_order_buffer)
            video_dict = {}  
    cache_name = '%s/%d.pkl'%(cache_folder,ix/point)
    cache_key = '%d'%(ix/point)
    out_pkl = open(cache_name, 'wb')
    pickle.dump(video_dict, out_pkl)
    out_pkl.close()

    video_cache_dict[cache_key] = cache_name
    video_order_buffer = video_dict.keys()
    video_order.extend(video_order_buffer)
    video_order.sort()
    # print video_order
    print 'video_list:%s'%(video_list)
    print 'list example:'
    print total
    return [video_cache_dict,video_order]

class videoReadTrain(videoRead):

  def initialize(self):
    params = eval(self.param_str)
    Train_batch_size = int(params['Train_batch_size'])
    multylabel = bool(params['multylabel']=='True')
    deep = bool(params['deep']=='True')
    lomo = bool(params['lomo']=='True')
    lomo_dim = int(params['lomo_dim'])
    map_lomo = {}
    map_lomo['flag'] =  bool(params['map_lomo']=='True')
    map_lomo['block_size'] =  int(params['block_size'])
    map_lomo['block_step'] =  int(params['block_step'])
    map_lomo['bin_size'] =  int(params['bin_size'])
    map_lomo['pad_size'] =  int(params['pad_size'])
    map_lomo['tau'] =  0.3
    map_lomo['R'] = 5
    map_lomo['numPoints'] = 4
    map_lomo['RGB_para'] = [bool(params['RGB']=='True'),8]
    map_lomo['HSV_para'] = [bool(params['HSV']=='True'),8]
    map_lomo['SILTP_para'] = [bool(params['SILTP']=='True'),16]
    map_lomo['data_dir'] = params['data_dir']
    map_lomo['lomo_dir'] = params['lomo_dir']
    
    chnnels = int(params['channels'])
    height = int(params['height'])
    width = int(params['width'])
    path_root = params['path_root']
    
    TrainVideolist = params['file_list']
    

    self.train_or_test = 'train'
    self.buffer_size = Train_batch_size  #num videos processed per batch
    self.N = self.buffer_size
    self.idx = 0
    self.channels = chnnels
    self.height = height
    self.width = width
    self.path_to_images = path_root  # the pre path to the datasets
    # video_dict = readlistFromFile(TrainVideolist,self.height,self.width)
    video_dict = readlistFromFileToCache(TrainVideolist,self.height,self.width)
    # self.video_dict = video_dict[0]
    self.video_cache_dict = video_dict[0]
    self.video_order = video_dict[1]
    self.num_videos = len(self.video_order)
    self.multylabel = multylabel
    self.deep = deep
    self.lomo = lomo
    self.lomo_dim = lomo_dim
    self.map_lomo = map_lomo

class videoReadTest(videoRead):

  def initialize(self):
    params = eval(self.param_str)
    Test_batch_size = int(params['Test_batch_size'])
    multylabel = bool(params['multylabel']=='True')
    deep = bool(params['deep']=='True')
    lomo = bool(params['lomo']=='True')
    lomo_dim = int(params['lomo_dim'])
    map_lomo = {}
    map_lomo['flag'] =  bool(params['map_lomo']=='True')
    map_lomo['block_size'] =  int(params['block_size'])
    map_lomo['block_step'] =  int(params['block_step'])
    map_lomo['bin_size'] =  int(params['bin_size'])
    map_lomo['pad_size'] =  int(params['pad_size'])
    map_lomo['tau'] =  0.3
    map_lomo['R'] = 5
    map_lomo['numPoints'] = 4

    map_lomo['RGB_para'] = [bool(params['RGB']=='True'),8]
    map_lomo['HSV_para'] = [bool(params['HSV']=='True'),8]
    map_lomo['SILTP_para'] = [bool(params['SILTP']=='True'),16]
    map_lomo['data_dir'] = params['data_dir']
    map_lomo['lomo_dir'] = params['lomo_dir']
    
    chnnels = int(params['channels'])
    height = int(params['height'])
    width = int(params['width'])
    path_root = params['path_root']
    
    TestVideolist = params['file_list']
    
    #print multylabel
    self.train_or_test = 'train'
    self.buffer_size = Test_batch_size  #num videos processed per batch
    self.N = self.buffer_size
    self.idx = 0
    self.channels = chnnels
    self.height = height
    self.width = width
    self.path_to_images = path_root  # the pre path to the datasets
    # video_dict = readlistFromFile(TestVideolist,self.height,self.width)
    video_dict = readlistFromFileToCache(TestVideolist,self.height,self.width)
    # self.video_dict = video_dict[0]
    self.video_cache_dict = video_dict[0]
    self.video_order = video_dict[1]
    self.num_videos = len(self.video_order)
    self.multylabel = multylabel
    self.deep = deep
    self.lomo = lomo
    self.lomo_dim = lomo_dim
    self.map_lomo = map_lomo

