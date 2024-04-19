# Import private modules
import src.config as config
import src.utils as utils

# Import public modules
import os
import re
import sys
import pdb
import nrrd
import time
import tqdm
import torch
import psutil
import functools
import traceback
import torchvision
import numpy as np
from pathlib import Path
import torch.multiprocessing as torchMP
torchMP.set_start_method('spawn', force=True) # spawn is default in windows, fork is default in linux
print = functools.partial(print, flush=True)

import warnings
warnings.filterwarnings('error')

class PointAndScribbleDataloader:

    def __init__(self, dirParams, sliceParams, dataloaderParams, show=False, verbose=False) -> None:
        
        # Step 0 - Set parameters
        self.dirParams = dirParams
        self.sliceParams = sliceParams
        self.dataloaderParams = dataloaderParams
        self.show = show
        self.verbose = verbose

        # Step 1 - Debug vars
        self.patientProcessPID = None
        self.annotationProcessesPIDs = []

        # Step 2 - Init funcs
        self._initSharedVars()
        self._initWorkers()

    def _initSharedVars(self):

        # Step 1 - Initialize worker queues 
        self.workerCount            = self.dataloaderParams[config.KEY_ANNOTATIONLOADER][config.KEY_ANNOTATIONLOADER_WORKERS]
        self.workerInputQueues = [torchMP.Queue() for _ in range(self.workerCount)]
        for workerId in range(self.workerCount):
            self.workerInputQueues[workerId].cancel_join_thread() # we dont want workerInputQueues to stop the main process from exiting
        self.workerOutputQueue = torchMP.Queue()
        self.workerProcesses   = []
        self.workerProcessesPIDs = []

        # Step 2 - Initialize shared variables
        self.workerStopEvent = torchMP.Event()
        self.mpLock    = torchMP.Lock()
    
    def _initWorkers(self):

        # Step 0 - Init
        self.workerCount = self.dataloaderParams[config.KEY_ANNOTATIONLOADER][config.KEY_ANNOTATIONLOADER_WORKERS]
        self.patientPaths, self.patientIdxToNameObj = utils.getPatientNamesAndPaths(self.dirParams)
        
        # Step 1 - Start patient workers
        self.annotationProcesses = []

        # Trial 1 - Individual process launch (works)
        if 1:
            for workerId in range(self.workerCount):
                p = torchMP.Process(target=getAnnotation, args=(workerId
                                        , self.workerInputQueues[workerId], self.workerOutputQueue
                                        , self.workerStopEvent
                                        , self.sliceParams, self.dataloaderParams
                                        , self.patientPaths, self.patientIdxToNameObj)
                                    , daemon=True)
                p.start()
                self.workerProcesses.append(p) # only add a worker to self.workerProcesses list after it has started
                self.workerProcessesPIDs.append(p.pid)
        
        # Trial 2 - Pool launch (failed)
        elif 0:
            self.manager = torchMP.Manager()
            self.pool    = torchMP.Pool(processes=self.workerCount)
            self.workerInputQueues = [self.manager.Queue() for _ in range(self.workerCount)]
            self.workerOutputQueue = self.manager.Queue()

            for workerId in range(self.workerCount):
                print (' - [PointAndScribbleDataloader._initWorkers()] Adding process to pool')
                self.pool.apply_async(getAnnotation, args=(workerId
                                        , self.workerInputQueues[workerId], self.workerOutputQueue
                                        , self.workerStopEvent
                                        , self.sliceParams, self.dataloaderParams
                                        , self.patientPaths, self.patientIdxToNameObj))
            print (' - [PointAndScribbleDataloader._initWorkers()] All processes added to pool')
        
    def __len__(self):

        dataloaderLen = len(self.patientPaths) * self.sliceParams[config.KEY_PERVIEW_SLICES] * 3
        # if 1: dataloaderLen = 40; print (' - [PointAndScribbleDataloader.__len__()] dataloader length is set to 40')

        if dataloaderLen == 0:
            print (' - [PointAndScribbleDataloader.__len__()] dataloader length is 0. Exiting...')
            sys.exit(0)

        return dataloaderLen

    def fillInputQueues(self):

        try:
            
            # Step 0 - Check input queue lengths
            for workerId in range(self.workerCount):
                if not self.workerInputQueues[workerId].empty():
                    print (' - [PointAndScribbleDataloader.fillInputQueues()] Worker {} still has {} items in input queue'.format(workerId, self.workerInputQueues[workerId].qsize()))
                    self.emptyAllQueues()

            # Step 1 - Get Data
            patientIdxToSliceIdxsObjs = utils.generatePatientSlicesIndexes(self.patientIdxToNameObj, self.workerCount, self.sliceParams)

            # Step 2 - Fill queues
            for workerId in range(self.workerCount):
                for patientIdx in patientIdxToSliceIdxsObjs[workerId]:
                    sliceIdxs = list(patientIdxToSliceIdxsObjs[workerId][patientIdx])
                    for sliceIdx in sliceIdxs:
                        self.workerInputQueues[workerId].put((patientIdx, sliceIdx))

        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def emptyAllQueues(self):

        try:
            
            for workerId in range(self.workerCount):
                while not self.workerInputQueues[workerId].empty():
                    try: _ = self.workerInputQueues[workerId].get_nowait()
                    except self.workerInputQueues[workerId].Empty: break

            while not self.workerOutputQueue.empty():
                try: _ = self.workerOutputQueue.get_nowait()
                except self.workerOutputQueue.Empty: break
            
            print (' - [PointAndScribbleDataloader.emptyAllQueues()] workerInputQueues: {} || workerOutputQueue: {}'.format(
                [self.workerInputQueues[i].qsize() for i in range(self.workerCount)], self.workerOutputQueue.qsize())
            )

        except:
            traceback.print_exc()
            pdb.set_trace()

    # Main loop
    def __iter__(self):
        
       
        try:

            # Step - 0 Init
            self.batchSize = self.dataloaderParams[config.KEY_ANNOTATIONLOADER][config.KEY_BATCH_SIZE]
            batchCT, batchPT, batchGT, batchPred, batchZ1, batchZ2, batchMeta = [], [], [], [], [], [], []
            device = self.dataloaderParams[config.KEY_ANNOTATIONLOADER][config.KEY_DEVICE]
            
            # Step 1 - Start epoch
            self.fillInputQueues()
            
            while True:
                if not self.workerOutputQueue.empty():

                    # Step 2.1 - Get data point                
                    try:
                        args = self.workerOutputQueue.get(timeout=config.QUEUE_TIMEOUT)
                        xCT, xPT, yGT, yPred, z1, z2, meta = args # xCT = [H,W,D,1]    
                    except self.workerOutputQueue.Empty: # this is the same as the if condition above, except that you can specify a timeout
                        continue
                    
                    # Step 2.2 - Append to batch
                    if len(batchCT) < self.batchSize:
                        batchCT.append(xCT)
                        batchPT.append(xPT)
                        batchGT.append(yGT)
                        batchPred.append(yPred)
                        batchZ1.append(z1)
                        batchZ2.append(z2)
                        batchMeta.append(meta)
                    
                    # Step 2.3 - Make batch
                    if len(batchCT) == self.batchSize:
                        
                        batchCT = collate_tensor_fn(batchCT) # [B,H,W,D,1]
                        batchPT = collate_tensor_fn(batchPT)
                        batchGT = collate_tensor_fn(batchGT)
                        batchPred = collate_tensor_fn(batchPred)
                        batchZ1 = collate_tensor_fn(batchZ1)
                        batchZ2 = collate_tensor_fn(batchZ2)
                        batchMeta = np.vstack(batchMeta).T

                        yield batchCT, batchPT, batchGT, batchPred, batchZ1, batchZ2, batchMeta

                        if self.show:
                            utils.showFunc(batchCT, batchPT, batchGT, batchPred, batchZ1, batchZ2, batchMeta)
                            pdb.set_trace()
                        
                        batchCT, batchPT, batchGT, batchPred, batchZ1, batchZ2, batchMeta = [], [], [], [], [], [], []
                    
                    # Step 3 - End condition (if all input and output queue are empty)
                    if np.all([self.workerInputQueues[i].empty() for i in range(self.workerCount)]) and self.workerOutputQueue.empty():
                        break    
                    
        except GeneratorExit:
            print (' - [PointAndScribbleDataloader.__iter__()] GeneratorExit')
            self.emptyAllQueues()
            return 
        
        except KeyboardInterrupt:
            print(" - [PointAndScribbleDataloader.__iter__()] function was interrupted.")
            self.closeProcesses()

        except:
            traceback.print_exc()
            pdb.set_trace()

    def closeProcesses(self):
        
        try:
            # Step 1 - Set stop event
            with self.mpLock:
                self.workerStopEvent.set()  # this should break the while loop in all workers
            
            # Step 2 - Join all workers
            # [TODO: Should I empty all queues before joining?]
            for workerId in range(len(self.workerProcesses)):
                self.workerProcesses[workerId].join()

            # Step 3 - Close all queues
            for workerId in range(self.workerCount):
                self.workerInputQueues[workerId].cancel_join_thread() # The cancel_join_thread() method is used to prevent the background thread associated with a queue from joining the main thread when the program exits. By default, when a program terminates, it waits for all non-daemon threads to complete before exiting.
                self.workerInputQueues[workerId].close()
            self.workerOutputQueue.cancel_join_thread()
            self.workerOutputQueue.close()
        finally:
            for workerId in range(self.workerCount):
                if self.workerProcesses[workerId].is_alive():
                    print (' - [PointAndScribbleDataloader.closeProcesses()] Worker {} is still alive. Terminating...'.format(workerId))
                    self.workerProcesses[workerId].terminate()

def getAnnotation(workerId, mpInputQueue, mpOutputQueue, mpStopEvent, sliceParams, dataloaderParams, patientPaths, patientIdxToNameObj):
    """
    Params
    ------
    workerId : int
        Worker id
    mpInputQueue : torchMP.Queue
        Input queue (containing (patientIdx, sliceIdx) tuples)
    mpOutputQueue : torchMP.Queue
        Output queue (will contain (xCT, xPT, yGT, yPred, z1, z2, meta) tuples)
    mpStopEvent : torchMP.Event
        Stop event
    sliceParams : dict
        Slice parameters
    dataloaderParams : dict
        Dataloader parameters
    patientPaths : dict
        Patient paths
    patientIdxToNameObj : dict
        Patient index to name object
    """

    try:
        
        # Step 0 - Init
        maxQueueLength = dataloaderParams[config.KEY_ANNOTATIONLOADER][config.KEY_QUEUE_MAXLEN]
        currentPatientObj = {}
        torch.set_num_threads(1) # [NOTE: setting > 1 causes slow down of dataloader. Why??]
        while not mpStopEvent.is_set():

            # Step 0.1 - Init
            if not mpInputQueue.empty() and mpOutputQueue.qsize() < maxQueueLength:
                
                # Step 1 - Get data from input queue
                patientIdx, sliceIdx = mpInputQueue.get() 
                # time.sleep(1)
                patientName = patientIdxToNameObj[patientIdx]
                # print (' - [PointAndScribbleDataset._getPatientData()] workerId: {} || patientName: {} || sliceIdx: {}'.format(workerId, patientName, sliceIdx))

                # Step 2 - Check if patientIdx data exists. If not, load it
                if patientName not in currentPatientObj:
                    
                    # Step 2.1 - Reset currentPatientObj
                    currentPatientObj = {patientName: {
                        config.KEY_CT: None, config.KEY_PET: None, config.KEY_GT: None, config.KEY_PRED: None
                        , config.KEY_FAILURE_AREAS_FP: None, config.KEY_FAILURE_AREAS_FN: None, config.KEY_SORTED_AXIAL: None, config.KEY_SORTED_SAGITTAL: None, config.KEY_SORTED_CORONAL: None
                    }}

                    # Step 2.2 - Get patient data
                    thisPatientPaths = patientPaths[patientName]
                    xCT, xPET, yGT, yPred, errorFP, errorFN, idsSortedAxial, idsSortedSagittal, idsSortedCoronal = utils.getPatientData(thisPatientPaths, sliceParams, patientName)
                    currentPatientObj[patientName][config.KEY_CT]   = torch.unsqueeze(torch.as_tensor(xCT), -1) # [H,W,D,1] # as_tensor avoid memory copy 
                    currentPatientObj[patientName][config.KEY_PET]  = torch.unsqueeze(torch.as_tensor(xPET), -1)
                    currentPatientObj[patientName][config.KEY_GT]   = torch.unsqueeze(torch.as_tensor(yGT), -1)
                    currentPatientObj[patientName][config.KEY_PRED] = torch.unsqueeze(torch.as_tensor(yPred), -1)
                    currentPatientObj[patientName][config.KEY_FAILURE_AREAS_FP] = errorFP
                    currentPatientObj[patientName][config.KEY_FAILURE_AREAS_FN] = errorFN
                    currentPatientObj[patientName][config.KEY_SORTED_AXIAL]     = idsSortedAxial
                    currentPatientObj[patientName][config.KEY_SORTED_SAGITTAL]  = idsSortedSagittal
                    currentPatientObj[patientName][config.KEY_SORTED_CORONAL]   = idsSortedCoronal
                
                # Step 3 - Get annotations
                xCT   = currentPatientObj[patientName][config.KEY_CT]
                xPET  = currentPatientObj[patientName][config.KEY_PET]
                yGT   = currentPatientObj[patientName][config.KEY_GT]
                yPred = currentPatientObj[patientName][config.KEY_PRED]
                errorFP = currentPatientObj[patientName][config.KEY_FAILURE_AREAS_FP]
                errorFN = currentPatientObj[patientName][config.KEY_FAILURE_AREAS_FN]
                idsSortedAxial    = currentPatientObj[patientName][config.KEY_SORTED_AXIAL]
                idsSortedSagittal = currentPatientObj[patientName][config.KEY_SORTED_SAGITTAL]
                idsSortedCoronal  = currentPatientObj[patientName][config.KEY_SORTED_CORONAL]

                meta, interactionClass, interactionDistanceMapVolume = utils.getDistMapVolumeForViews((patientName, patientIdx), sliceIdx, errorFP, errorFN,
                                                                        idsSortedAxial, idsSortedSagittal, idsSortedCoronal, sliceParams)
                meta = np.array(meta, dtype=object)

                # Step 4 - Prepare data for queue
                if interactionDistanceMapVolume is not None:
                    x1 = currentPatientObj[patientName][config.KEY_CT] # already a torch tensor
                    x2 = currentPatientObj[patientName][config.KEY_PET]
                    y1 = currentPatientObj[patientName][config.KEY_GT]
                    y2 = currentPatientObj[patientName][config.KEY_PRED]
                    if interactionClass == config.KEY_INTERACTION_FGD:
                        z1 = torch.unsqueeze(torch.as_tensor(interactionDistanceMapVolume), -1)
                        z2 = torch.zeros_like(z1)
                    elif interactionClass == config.KEY_INTERACTION_BGD:
                        z2 = torch.unsqueeze(torch.as_tensor(interactionDistanceMapVolume), -1)
                        z1 = torch.zeros_like(z2)
                
                else:
                    # viewStr = 'Axial' if axialBool else 'Sagittal' if sagittalBool else 'Coronal'
                    print (' - [PointAndScribbleDataset._getPatientData()] interactionDistanceMapVolume is None || requestedPatient: {} || requestedSliceId: {}, view: {}'.format(patientName, sliceIdx))
                    x1, x2, y1, y2, z1, z2 = None, None, None, None, None, None
                
                # Step 5 - Put data in output queue
                mpOutputQueue.put((x1, x2, y1, y2, z1, z2, meta))
                del x1, x2, y1, y2, z1, z2, meta, interactionClass, interactionDistanceMapVolume # save memory
        
    except KeyboardInterrupt:
        print("\n - [PointAndScribbleDataloader.getAnnotations()][worker={}] function was interrupted.".format(workerId))

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    finally:
        if mpStopEvent.is_set():
            mpOutputQueue.cancel_join_thread()
            mpOutputQueue.close()
        print (' - [PointAndScribbleDataloader.getAnnotations()][worker={}] Closed! '.format(workerId))

def collate_tensor_fn(batch):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)

if __name__ == "__main__":
    for workers_nb in [1,2,4,8]:
        for bs in [1,2,4,8]:
            print(f"workers: {workers_nb}, batch size: {bs} ---------start--------")
            
    
            DIR_FILE = Path(__file__).resolve().parent.absolute() # ./src
            DIR_ROOT = DIR_FILE.parent.absolute() # ./
            DIR_DATA = DIR_ROOT / '_data'  

            # Step 1 - Dataloader Params
            if 1:

                # Params Set 1
                if 1:

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

                    dataloaderParams = {
                        config.KEY_ANNOTATIONLOADER: {    
                            config.KEY_ANNOTATIONLOADER_WORKERS: workers_nb
                            , config.KEY_EPOCHS: 6 # [1,2]
                            , config.KEY_BATCH_SIZE:bs
                            , config.KEY_QUEUE_MAXLEN: 30
                            , config.KEY_DEVICE: config.deviceGPU # [config.deviceGPU, config.deviceCPU]
                        }
                    }
            
            # Step 2 - Dataloader loop
            if 1:

                # from viztracer import VizTracer
                # tracer = VizTracer()
                # tracer.start()
                dataloader = None
                try:
                    
                    dataloader = PointAndScribbleDataloader(dirParams, sliceParams, dataloaderParams, show=False, verbose=False)
                    epochs = dataloaderParams[config.KEY_ANNOTATIONLOADER][config.KEY_EPOCHS]
                    for epochId in range(epochs):
                        print ('\n ---------------------------------------------- [epochId={}]'.format(epochId))
                        t1 = time.time()
                        with tqdm.tqdm(total=len(dataloader)) as pbar:
                            counter = 0
                            for (xCT, xPT, yGT, yPred, zFgd, zBgd, meta) in dataloader:

                                pbar.update(dataloader.batchSize)
                                counter += dataloader.batchSize
                                # pdb.set_trace()

                                # if counter > 60:
                                #     print (' - [main()] Breaking after 10 iterations')
                                #     break
                        t2 = time.time()
                        dt = t2-t1 
                        print(f"time cost: {dt}, speed: {855/dt}")
                        dt_all['workers'].append(workers_nb)
                        dt_all['batch_size'].append(bs)
                        dt_all['cost_time'].append(dt)
                        dt_all['speed'].append(855/dt)
                    print ('\n - [main()] Closing dataloader')
                    dataloader.closeProcesses()
                    print (' - [main()] Closed dataloader')
                  
          
                except KeyboardInterrupt:
                    # tracer.save()
                    if dataloader is not None: dataloader.closeProcesses()
                
                except:
                    traceback.print_exc()
                    pdb.set_trace()

    df = pd.DataFrame(dt_all)
    df.to_csv('dataloader1_table.csv', index=False)  # index=False 表示不保存索引
    print(f"finishe all!")
