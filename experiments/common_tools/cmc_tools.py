def format_time(num_time):
    seconds=num_time%60
    minuites=num_time/60%60
    hours=num_time/3600%24
    days=num_time/3600/24
    fm_time={'sec':seconds,'min':minuites,'hours':hours,'days':days}
    return fm_time
def generatePredictList_Batch(net, data_operator, probes_list, gallerys_list,test_frames,test_buffer):
    import time
    import numpy as np
    predictLists = []
    probes = probes_list[0]
    gallerys = gallerys_list[0]
    
    key_probe = probes_list[1]
    key_gallerys = gallerys_list[1]
    data_operator.setup()
    c = 0
    start_time = time.time()
    total_time = 0
    total_iter = int((len(key_probe))*(len(key_gallerys))/test_buffer)+1
    predict_totalList = []
    for i in range(total_iter ):
        data_operator.forward(net)
        net.forward()
        outScore=net.blobs['softmax_score'].data.reshape((test_frames*test_buffer,2))
        similarScore = []
        similarScore=outScore[(test_frames-1)*test_buffer:,1]
        predict_totalList.extend(similarScore)
        if i% 1000 == 0:
            end_time = time.time()
            using_time = end_time - start_time
            total_time = total_time + using_time
            start_time = end_time
            time1 = format_time(long(total_time))
            print 'Time ellipsing: %d days %d hours %d mins %d seconds \n' % (
                time1['days'], time1['hours'], time1['min'], time1['sec'])
            print '%d batch probes has been tested (batch is %d)' % (i,test_buffer)
    print "%d pairs has finished"%(len(predict_totalList))
    for key in key_probe:
        predict = []

        for keys in key_gallerys:
            if c >= len(predict_totalList):
                break
            predict.append(predict_totalList[c])
            c += 1
        predictRanklist=np.argsort(predict)[::-1]
        #print predictRanklist
        predictLists.append(predictRanklist)

        #if c == 1:
        #    break
    #print predictLists
    end_time = time.time()
    using_time = end_time - start_time
    total_time = total_time + using_time
    start_time = end_time
    time1 = format_time(long(total_time))
    print 'Time ellipsing: %d days %d hours %d mins %d seconds \n' % (
        time1['days'], time1['hours'], time1['min'], time1['sec'])
    #print('\r  Processing %dx%d pairs cost %f second time'%(len(probes),len(gallerys),(finish-start)))
    return predictLists #single and last
def generatePredictList_Batch_seq(net, data_operator, probes_list, gallerys_list,test_frames,test_buffer):
    import time
    import numpy as np
    probes = probes_list[0]
    gallerys = gallerys_list[0]
    
    key_probe = probes_list[1]
    key_gallerys = gallerys_list[1]

    predictLists = []
    predictLists_mean = []
    predictLists_max = []
    
    data_operator.setup()
    c = 0
    start_time = time.time()
    total_time = 0
    total_iter = int((len(key_probe))*(len(key_gallerys))/test_buffer)+1
    predict_totalList = []
    predict_totalList_max = []
    for i in range(total_iter ):
        data_operator.forward(net)
        net.forward()
        outScore=net.blobs['softmax_score'].data.reshape((test_frames*test_buffer,2))
        similarScore = []
        similarScore_mean = []
        similarScore_max = []
        similarScore.append(outScore[(test_frames-1)*test_buffer:,1])
        for idx in range(test_buffer):
            similarScore_mean.append(sum(outScore[idx::test_buffer,1])/test_frames)
            similarScore_max.append(max(outScore[idx::test_buffer,1]))
        predict_totalList.extend(similarScore)
        predict_totalList_mean.extend(similarScore_mean)
        predict_totalList_max.extend(similarScore_max)
        if i% 1000 == 0:
            end_time = time.time()
            using_time = end_time - start_time
            total_time = total_time + using_time
            start_time = end_time
            time1 = format_time(long(total_time))
            print 'Time ellipsing: %d days %d hours %d mins %d seconds \n' % (
                time1['days'], time1['hours'], time1['min'], time1['sec'])
            print '%d batch probes has been tested (batch is %d)' % (i,test_buffer)
    print "%d pairs has finished"%(len(predict_totalList))
    for key in key_probe:
        predict = []

        for keys in key_gallerys:
            if c >= len(predict_totalList):
                break
            predict.append(predict_totalList[c])
            c += 1
        predictRanklist=np.argsort(predict)[::-1]
        #print predictRanklist
        predictLists.append(predictRanklist)
    c = 0
    for key in key_probe:
        predict_mean = []

        for keys in key_gallerys:
            if c >= len(predict_totalList_mean):
                break
            predict_mean.append(predict_totalList_mean[c])
            c += 1
        predictRanklist_mean=np.argsort(predict_mean)[::-1]
        #print predictRanklist_max
        predictLists_mean.append(predictRanklist_mean)
        #if c == 1:
        #    break
    c = 0
    for key in key_probe:
        predict_max = []

        for keys in key_gallerys:
            if c >= len(predict_totalList_max):
                break
            predict_max.append(predict_totalList_max[c])
            c += 1
        predictRanklist_max=np.argsort(predict_max)[::-1]
        #print predictRanklist_max
        predictLists_max.append(predictRanklist_max)
        #if c == 1:
        #    break
    #print predictLists
    end_time = time.time()
    using_time = end_time - start_time
    total_time = total_time + using_time
    start_time = end_time
    time1 = format_time(long(total_time))
    print 'Time ellipsing: %d days %d hours %d mins %d seconds \n' % (
        time1['days'], time1['hours'], time1['min'], time1['sec'])
    return [predictLists,predictLists_mean,predictLists_max]

def getCVPRcmc():
    #return the cmc values, 100 dim vetor
    import numpy as np
    cmcIndex=[0,4,8,12,16,21,25,29,33,37,41,45,49,53]
    cmcOfCVPRImproved=[0.5474,0.8753,0.9293,0.9712,0.9764,0.9811,0.9899,0.9901,0.9912,0.9922,0.9937,0.9945,0.9951,1]
    pOfCVPRImproved = np.poly1d(np.polyfit(cmcIndex,cmcOfCVPRImproved,10))
    x_line=range(50)
    cmc=pOfCVPRImproved(x_line)
    return cmc
def evaluateCMC( predictLists, key_probes, key_gallerys):
    import numpy as np
    N=len(key_probes)
    R=len(predictLists[0])
    histogram=np.zeros(N)
    for testIdx in range(N):
        for rankIdx in range(R):
            histogram[rankIdx]+=1*(key_gallerys[int(predictLists[testIdx][rankIdx])]==key_probes[testIdx])    #1*(true or false)=1 or 0
    cmc=np.cumsum(histogram)
    return cmc/N
