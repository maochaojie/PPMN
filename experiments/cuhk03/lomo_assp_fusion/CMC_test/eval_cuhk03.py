import os
root = '../../../../'

os.chdir(root)
#from experiments.common_tools.cmc import evaluateCMC
print 'current work dir : %s'%os.getcwd()
caffe_root ='caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
commontool_root = 'experiments_py_lomo/common_tools'
sys.path.insert(0,commontool_root)
from lomo_map import *

def readList(list_name): 
    import random
    import os
    file_object = open(list_name)
    try:
        all_the_text = file_object.read()
    finally:
        file_object.close()

    lines = all_the_text.split('\n')
    #print all_the_text
    DATA_DIR='dataset/cuhk03/'
    probes=[]
    gallerys=[]
    for filename in lines:
        if filename!='' :
            campair_no = int(filename.split(',')[0])
            person_id = int(filename.split(',')[1])
            #print int(campair_no)
            while True:
                probe_no=random.randint(1,5)
                probe_filename = DATA_DIR +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,probe_no)
                if os.path.isfile(probe_filename):
                    probes.append(probe_filename)
                    break
            while True:
                gallery_no=random.randint(6,10)  
                gallery_filename = DATA_DIR +  'data/campair_%d/'%campair_no + '%02d_%04d_%02d.jpg'%(campair_no,person_id,gallery_no)
                if os.path.isfile(gallery_filename):
                    gallerys.append(gallery_filename)
                    break
    if len(probes)!=len(gallerys):
        print('something wrong! list length does not match!/n')
        return 0
    else:
        return probes,gallerys

def generateScoreList(net,probes,gallerys,map_lomo, deep = True):
    if deep==True:
        transformer = caffe.io.Transformer({'data': (net.blobs['data'].data.shape)})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.array([ 104,  117,  123])) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        N,C,H,W=net.blobs['data'].data.shape
    if map_lomo['flag'] == True:
        extractor = feature_extractor(RGB_para = map_lomo['RGB_para'],HSV_para=map_lomo['HSV_para'],SILTP_para=map_lomo['SILTP_para'],\
                                          block_size = map_lomo['block_size'], block_step = map_lomo['block_step'], pad_size = map_lomo['pad_size'], \
                                          tau = map_lomo['tau'], R = map_lomo['R'], numPoints = map_lomo['numPoints'])
    scoreList=[]

    from time import clock
    start=clock()
    #galleryData is same for each probe
    galleryLen=len(gallerys)
    galleryImgList=[]
    galleryLomoList=[]
    check = True
    for galleryIdx in range(galleryLen):
        galleryName=gallerys[galleryIdx]
        if map_lomo['flag'] == True:
            map_lomo_path = galleryName.replace(map_lomo['data_dir'],map_lomo['lomo_dir'])[:-4]+'.npy'
            # print map_lomo_path
            if os.path.isfile(map_lomo_path):
                map_lomo_fea = np.load(map_lomo_path)
            else:
                map_lomo_fea = extractor.extract_feature(galleryName)
                map_lomo_folder = map_lomo_path.replace(map_lomo_path.split('/')[-1],'')
                if not os.path.exists(map_lomo_folder):
                    os.makedirs(map_lomo_folder)
                np.save(map_lomo_path, map_lomo_fea)
            if check:
                C_lomo,H_lomo,W_lomo = map_lomo_fea.shape
                check = False
        data_in = np.zeros((1,1))
        if deep == True:
            data_in = transformer.preprocess('data', caffe.io.load_image(galleryName))
        galleryLomoList.append(map_lomo_fea)
        galleryImgList.append(data_in)
        galleryIdx+=1
    print [C_lomo,H_lomo,W_lomo]
    #galleryData and probeData
    if map_lomo['flag'] == True:
        galleryLomo=np.asarray(galleryLomoList)
        probeLomo=np.zeros((galleryLen,C_lomo,H_lomo,W_lomo))
        net.blobs['data_map_lomo'].reshape(galleryLen,C_lomo,H_lomo,W_lomo)
        net.blobs['data_map_lomo_p'].reshape(galleryLen,C_lomo,H_lomo,W_lomo)
    if deep == True:
        galleryImg=np.asarray(galleryImgList)
        probeImg=np.zeros((galleryLen,C,H,W))
        net.blobs['data'].reshape(galleryLen,C,H,W)
        net.blobs['data_p'].reshape(galleryLen,C,H,W)

    print galleryLomo.shape


    #process each probe
    for probeIdx in range(len(probes)):
        probeName=probes[probeIdx]
        probeData = {}
        if map_lomo['flag'] == True:
            map_lomo_path = probeName.replace(map_lomo['data_dir'],map_lomo['lomo_dir'])[:-4]+'.npy'
            if os.path.isfile(map_lomo_path):
                map_lomo_fea = np.load(map_lomo_path)
            else:
                map_lomo_fea = extractor.extract_feature(probeName)
                map_lomo_folder = map_lomo_path.replace(map_lomo_path.split('/')[-1],'')
                if not os.path.exists(map_lomo_folder):
                    os.makedirs(map_lomo_folder)
                np.save(map_lomo_path, map_lomo_fea)
            if check:
                C_lomo,H_lomo,W_lomo = map_lomo_fea.shape
            probeData['map_lomo'] = map_lomo_fea
            probeLomo[:,:,:,:]=probeData['map_lomo']
            net.blobs['data_map_lomo'].data[:] = probeLomo
            net.blobs['data_map_lomo_p'].data[:] = galleryLomo
        if deep == True:
            probeData['img'] = transformer.preprocess('data', caffe.io.load_image(probeName))
            probeImg[:,:,:,:]=probeData['img']
            net.blobs['data'].data[:] = probeImg
            net.blobs['data_p'].data[:] = galleryImg
        outScore = []
        #batch data assignment
        
        
        #net forwad
        net.forward()
        #get output score
        # outScore.extends(net.blobs['softmax_score'].data[:,(0,1)])    #softmax_score[0] and softmax_score[1]
        
        outScore = net.blobs['softmax_score_flow'].data[:,(0,1)]#+net.blobs['softmax_score_lomo'].data[:,(0,1)]   #softmax_score[0] and softmax_score[1]
    # print outScore.shape
        score_sum=np.exp(outScore[:,0]*1.0)+np.exp(outScore[:,1]*1.0)
        similarScore=outScore[:,1]#np.exp(outScore[:,1]*1.0)/score_sum
        #scoreList.append each probe score
        scoreList.append(similarScore.tolist())
        if (probeIdx+1)%10==0:
            sys.stdout.write('\r%3d/%d, '%(probeIdx+1,len(probes))+probeName)
            sys.stdout.flush()
    #we get scoreList, then cal predictLists
    predictLists=[]
    for score in scoreList:
        probeRankList=np.argsort(score)[::-1]
        predictLists.append(probeRankList)
    finish=clock()
    print('\r  Processing %dx%d pairs cost %f second time'%(len(probes),len(gallerys),(finish-start)))
    return scoreList,predictLists

def calCMC(net,set_no,map_lomo,deep=True,rand_times=10):
    from cmc import evaluateCMC
    DATA_DIR= 'dataset/cuhk03/'
    list_name=DATA_DIR+'exp_set/set%02d_test_noval.txt'%(set_no)
    print list_name+'\n'
    #rand 10 times for stable result
    cmc_list=[]
    for i in range(rand_times):
        print 'Round %d with rand list:'%i
        probes,gallerys=readList(list_name)
        scoreList,predictLists=generateScoreList(net,probes,gallerys,map_lomo,deep=deep)
        gtLabels=range(len(probes))
        cmc=evaluateCMC(gtLabels,predictLists)
        cmc_list.append(cmc)
    return np.average(cmc_list,axis=0)

def getCVPRcmc():
    #return the cmc values, 100 dim vetor
    import numpy as np
    cmcIndex=[0,4,8,12,16,21,25,29,33,37,41,45,49,53]
    cmcOfCVPRImproved=[0.5474,0.8753,0.9293,0.9712,0.9764,0.9811,0.9899,0.9901,0.9912,0.9922,0.9937,0.9945,0.9951,1]
    pOfCVPRImproved = np.poly1d(np.polyfit(cmcIndex,cmcOfCVPRImproved,10))
    x_line=range(50)
    cmc=pOfCVPRImproved(x_line)
    return cmc

def plotCMC(cmcDict,pathname):
    import matplotlib.pyplot as plt
    #get_ipython().magic(u'matplotlib inline')   
    from matplotlib.legend_handler import HandlerLine2D
    import numpy as np

    #plot the cmc curve, record CVPR from the pdf paper.cmc[0,4,8,12,16,21,25,29,33,37,41,45,49]
    rank2show=25
    rankStep=1
    cmcIndex=np.arange(0,rank2show,rankStep)   #0,5,10,15,20,25

    colorList=['rv-','g^-','bs-','yp-','c*-','mv-','kd-','gs-','b^-']
    #start to plot
    plt.ioff()
    fig = plt.figure(figsize=(6,5),dpi=180)
    sortedCmcDict = sorted(cmcDict.items(), key=lambda (k, v): v[1])[::-1]
    for idx in range(len(sortedCmcDict)):
        cmc_dictList=sortedCmcDict[idx]
        cmc_name=cmc_dictList[0]
        cmc_list=cmc_dictList[1]
        #print cmc_name,": ",cmc_list
        #x for plot
        x_point=[item+1 for item in cmcIndex]
        x_line=range(rank2show)
        x_plot=[temp+1 for temp in x_line]
        #start plot
        plt.plot(x_plot, cmc_list[x_line],colorList[idx],label="%02.02f%% %s"%(100*cmc_list[0],cmc_name))
        plt.plot(x_point,cmc_list[cmcIndex],colorList[idx]+'.')
        #plt.legend(loc=4,handler_map={line: HandlerLine2D(numpoints=1)})
        #idx of color +1
        idx+=1
    #something to render

    plt.xlabel('Rank')
    plt.ylabel('Identification Rate')
    plt.xticks(np.arange(0,rank2show+1,5))
    plt.yticks(np.arange(0,1.01,0.1))
    plt.grid()
    plt.legend(loc=4)
    plt.savefig(pathname)
    #plt.show()

    #end of show
    
def main():
    test_list=[2] #use set 1-10 for test (total 20)
    caffe.set_device(int(sys.argv[1]))
    caffe.set_mode_gpu()
    map_lomo = {}
    map_lomo['flag'] = True
    map_lomo['data_dir'] = 'data/'
    map_lomo['lomo_dir'] = 'lomo_test/'
    map_lomo['block_size'] =  8
    map_lomo['block_step'] =  4
    map_lomo['bin_size'] =  8
    map_lomo['pad_size'] =  2
    map_lomo['tau'] =  0.3
    map_lomo['R'] = 5
    map_lomo['numPoints'] = 4
    map_lomo['RGB_para'] = [True, 8]
    map_lomo['HSV_para'] = [True, 8]
    map_lomo['SILTP_para'] = [True, 16]

    CMC_DIC = {}
    for iter_num in range(2000, 20000, 2000):
        cmc_list=[]
        for set_no in test_list:
            #init net
            MODEL_FILE = './experiments_py_lomo/cuhk03/lomo_assp_fusion/deploy_twoflow_fusion.prototxt'
            PRETRAINED = './models/lomo/cuhk03/lomo_assp_fusion/20000_set02_iter_%d.caffemodel'%iter_num#30000_set02_v2_all_iter_%d.caffemodel'%iter_num
            net = None
            net = caffe.Classifier(MODEL_FILE, PRETRAINED,caffe.TEST)
            #caculate CMC
            cmc=calCMC(net,set_no,map_lomo,deep=True,rand_times=50)
            cmc_list.append(cmc)
        cmc_all=np.average(cmc_list,axis=0)
        print('\nCMC from rank 1 to rank %d:'%(len(cmc_all)))
        print(cmc_all)
        
        CMC_DIC['single_yaqing_%d'%iter_num] = cmc_all
        
    save_root = 'experiments_py_lomo/cuhk03/lomo_assp_fusion/CMC_test/'
    save_mat_path = save_root + 'cuhk03_set02_yqCMC_100round_20000_fusion.mat'
    import scipy.io as sio
    sio.savemat(save_mat_path,{'ours':CMC_DIC})

    save_path = save_root + 'cuhk03_set02_yqCMC_100round_20000_fusion.png'
    plotCMC(CMC_DIC, save_path)
    
if __name__ == '__main__':
    main()
