# Import private libs
import src.datasetBasic as datasetBasic
import src.dataloaderNew as dataloaderNew
import src.config as config

# Import public libs
import pdb
import copy
import time
import tqdm
import pprint
import traceback
import numpy as np
from pathlib import Path

import torch
import torchvision

def calculateTime(datasetLen, dataloader, epochs, workerCountList=[], batchSizeList=[], device=config.deviceCPU):

    timeForEpochs = []

    try:

        # Step 1 - Loop over epochs
        print ('')
        for epoch in range(epochs):
            
            with tqdm.tqdm(total=datasetLen, desc=' - Epoch {}/{}'.format(epoch+1, epochs)) as pbar:
                t1 = time.time()
                
                # Step 2 - Loop over dataloader
                for i, (x1,x2,y1,y2,z1,z2,meta) in enumerate(dataloader):
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y1 = y1.to(device)
                    y2 = y2.to(device)
                    z1 = z1.to(device)
                    z2 = z2.to(device)

                    pbar.update(x1.shape[0])
                
                t2= time.time()
                timeForEpochs.append(t2-t1)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return timeForEpochs

if __name__ == "__main__":

    try:

        ############################################################### Step 0 - Declare global params
        DIR_FILE = Path(__file__).resolve().parent.absolute() # ./src
        DIR_ROOT = DIR_FILE.parent.absolute() # ./
        DIR_DATA = DIR_ROOT / '_data'  

        ############################################################### Step 1 - Declare dataset specific params
        dirParams = {
            config.KEY_DIR_DATA_OG: DIR_DATA / 'trial1',
            config.KEY_REGEX_CT: 'img1',
            config.KEY_REGEX_PET: 'img2',
            config.KEY_REGEX_GT: 'mask',
            config.KEY_REGEX_PRED: 'pred',
            config.KEY_EXT: config.EXT_NRRD,
            config.KEY_STRFMT_CT: 'nrrd_{}_{}{}'.format('{}', 'img1', config.EXT_NRRD),
            config.KEY_STRFMT_PET: 'nrrd_{}_{}{}'.format('{}', 'img2', config.EXT_NRRD),
            config.KEY_STRFMT_GT: 'nrrd_{}_{}{}'.format('{}', 'mask', config.EXT_NRRD),
            config.KEY_STRFMT_PRED: 'nrrd_{}_{}{}'.format('{}', 'maskpred', config.EXT_NRRD)
        }

        sliceParams = {
            config.KEY_PERVIEW_SLICES: 5,
            config.KEY_KSIZE_SEGFAILURE: (3,3,3),
            config.KEY_LABEL: 1,
            config.KEY_INTERACTION_TYPE: [config.KEY_INTERACTION_SCRIBBLES], # [config.KEY_INTERACTION_POINTS, config.KEY_INTERACTION_SCRIBBLES]
            config.KEY_SCRIBBLE_TYPE: [config.KEY_SCRIBBLE_MEDIAL_AXIS] # config.KEY_SCRIBBLE_RANDOM, config.KEY_SCRIBBLE_MEDIAL_AXIS
        }

        transform = torchvision.transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor()
        ])

        ############################################################### Step 2 - Declare results object and other variables
        resultsObj = {config.KEY_WORKERS: [], config.KEY_BATCH_SIZE: [], config.KEY_TIMELIST: [], config.KEY_ITERPERSEC: []}
        
        # workerCountList, batchSizeList, totalEpochs = [1, 2, 4, 8], [1, 2, 4, 8], 6
        workerCountList, batchSizeList, totalEpochs = [4], [4,8], 1
        device = config.deviceCPU # config.deviceGPU

        ############################################################## Step 2 - Test datasetBasic
        print (' \n =====================================================> Testing datasetBasic.py')
        dataloader1ResultsObj = copy.deepcopy(resultsObj)
        if 1:
            dataset1              = datasetBasic.PointAndScribbleDataset(dirParams, sliceParams, transform=transform)
            dataset1Len           = len(dataset1)
            for workerCount in workerCountList:
                dataloader1ResultsObj[config.KEY_WORKERS].append(workerCount)
                for batchSize in batchSizeList:
                    dataloader1ResultsObj[config.KEY_BATCH_SIZE].append(batchSize)

                    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=batchSize, num_workers=workerCount
                                                                , shuffle=False
                                                                , pin_memory=True, pin_memory_device=config.nameGPU
                                                            )
                    timeForEpochs = calculateTime(dataset1Len, dataloader1, totalEpochs, workerCountList, batchSizeList, device)
                    itersPerSec   = [dataset1Len / timeForEpoch for timeForEpoch in timeForEpochs]
                    dataloader1ResultsObj[config.KEY_TIMELIST].append(timeForEpochs)
                    dataloader1ResultsObj[config.KEY_ITERPERSEC].append(itersPerSec)

        ############################################################### Step 3 - Test datasetNew
        print (' \n =====================================================> Testing dataloaderNew.py')
        dataloader2ResultsObj = copy.deepcopy(resultsObj)
        if 0:
            for workerCount in workerCountList:
                dataloader2ResultsObj[config.KEY_WORKERS].append(workerCount)
                for batchSize in batchSizeList:
                    dataloader2ResultsObj[config.KEY_BATCH_SIZE].append(batchSize)

                    dataloader2Params = {
                            config.KEY_ANNOTATIONLOADER: {    
                                config.KEY_ANNOTATIONLOADER_WORKERS: workerCount
                                , config.KEY_BATCH_SIZE:batchSize
                                , config.KEY_QUEUE_MAXLEN: 30
                                , config.KEY_DEVICE: config.deviceCPU # [config.deviceGPU, config.deviceCPU]
                            }
                        }

                    dataloader2    = dataloaderNew.PointAndScribbleDataloader(dirParams, sliceParams, dataloader2Params)
                    dataloader2Len = len(dataloader2)
                    timeForEpochs  = calculateTime(dataloader2Len, dataloader2, totalEpochs, workerCountList, batchSizeList, device)
                    itersPerSec    = [dataloader2Len / timeForEpoch for timeForEpoch in timeForEpochs]
                    dataloader2ResultsObj[config.KEY_TIMELIST].append(timeForEpochs)
                    dataloader2ResultsObj[config.KEY_ITERPERSEC].append(itersPerSec)
        
        ############################################################### Step 4 - Print results
        if 1:
            pprint.pprint(dataloader1ResultsObj)
            pprint.pprint(dataloader2ResultsObj)
            totalExps = 0
            dataloader1Done, dataloader2Done = False, False
            if len(dataloader1ResultsObj[config.KEY_WORKERS]) and len(dataloader1ResultsObj[config.KEY_BATCH_SIZE]):
                totalExps = len(dataloader1ResultsObj[config.KEY_WORKERS]) * len(dataloader1ResultsObj[config.KEY_BATCH_SIZE])
                dataloader1Done = True
                if len(dataloader2ResultsObj[config.KEY_WORKERS]) and len(dataloader2ResultsObj[config.KEY_BATCH_SIZE]):
                    dataloader2Done = True
            else:
                if len(dataloader2ResultsObj[config.KEY_WORKERS]) and len(dataloader2ResultsObj[config.KEY_BATCH_SIZE]):
                    totalExps = len(dataloader2ResultsObj[config.KEY_WORKERS]) * len(dataloader2ResultsObj[config.KEY_BATCH_SIZE])
                    dataloader2Done = True
                
            if totalExps: 
                expCounter = 0
                xTickList  = []
                d1AvgItersPerSecList, d2AvgItersPerSecList = [], []
                for workerId, worker in enumerate(dataloader1ResultsObj[config.KEY_WORKERS]):
                    for batchId, batch in enumerate(dataloader1ResultsObj[config.KEY_BATCH_SIZE]):
                        xTickList.append('W{}-B{}'.format(worker, batch))
                        if dataloader1Done:
                            d1AvgItersPerSecList.append(np.mean(dataloader1ResultsObj[config.KEY_ITERPERSEC][expCounter]))
                        if dataloader2Done:
                            d2AvgItersPerSecList.append(np.mean(dataloader2ResultsObj[config.KEY_ITERPERSEC][expCounter]))
                        expCounter += 1
                
                if dataloader1Done or dataloader2Done:
                    import matplotlib.pyplot as plt 
                    if dataloader1Done:
                        plt.bar(xTickList, d1AvgItersPerSecList, color='b', label='datasetBasic')
                    if dataloader2Done:
                        plt.bar(xTickList, d2AvgItersPerSecList, color='r', label='dataloaderNew')
                    
                    plt.xlabel('Worker-Batch')
                    plt.legend()
                    plt.savefig('dataloaderCompare__W{}__B{}.png'.format('-'.join(map(str, workerCountList)), '-'.join(map(str, batchSizeList))))
                    plt.show(block=False)
            
            pdb.set_trace()
    
    except:
        traceback.print_exc()
        pdb.set_trace()

