import time
import tqdm
import torch
import traceback
import numpy as np
import torch.multiprocessing as torchMP

def getPatientSliceArray(patientName, sliceId):
    time.sleep(0.2)
    return torch.tensor(np.random.rand(1, 128, 128, 128))

##################################################
# myDataset
##################################################

class myDataset(torch.utils.data.Dataset):
    def __init__(self, patientSlicesList, slicesPerPatient):
        self.patientSlicesList = patientSlicesList
        self.slicesPerPatient = slicesPerPatient
        
    def __len__(self):
        return self.slicesPerPatient * len(self.patientSlicesList)

    def __getitem__(self, idx):
        
        # Step 0 - Init
        patientIdx  = idx // self.slicesPerPatient
        sliceIdx    = idx % self.slicesPerPatient
        patientName = list(self.patientSlicesList.keys())[patientIdx]
        sliceId     = self.patientSlicesList[patientName][sliceIdx]

        # Step 1 - Get patient slice array
        patientSliceArray = getPatientSliceArray(patientName, sliceId)

        return patientSliceArray, [patientName, sliceId]

##################################################
# myNewDataloader
##################################################

QUEUE_TIMEOUT = 5.0

class myNewDataloader:

    def __init__(self, patientSlicesList, slicesPerPatient, numWorkers, batchSize) -> None:
        
        self.patientSlicesList = patientSlicesList
        self.slicesPerPatient  = slicesPerPatient
        self.numWorkers        = numWorkers
        self.batchSize         = batchSize

        self._initWorkers()
    
    def _initWorkers(self):
        
        # Step 1 - Initialize vas
        self.workerProcesses    = []
        self.workerInputQueues = [torchMP.Queue() for _ in range(self.numWorkers)]
        self.workerOutputQueue = torchMP.Queue()
        self.workerStopEvent   = torchMP.Event() # used in getSlice() and self.closeProcesses()
        self.mpLock            = torchMP.Lock()  # used in self.closeProcesses()
    
        for workerId in range(self.numWorkers):
                p = torchMP.Process(target=getSlice, args=(workerId, self.workerInputQueues[workerId], self.workerOutputQueue
                                        , self.workerStopEvent)
                                    , daemon=True)
                p.start()
                self.workerProcesses.append(p)
    
    def __len__(self):
        return self.slicesPerPatient * len(self.patientSlicesList)

    def fillInputQueues(self):
        """
        This function allows to split patients and slices across workers
        """
        patientNames = list(self.patientSlicesList.keys())
        for workerId in range(self.numWorkers):
            startIdx = workerId * len(patientNames) // self.numWorkers
            endIdx   = (workerId + 1) * len(patientNames) // self.numWorkers
            # print (' - [myNewDataloader.fillInputQueues()] Worker {} will process patients {} to {}: {}'.format(workerId, startIdx, endIdx, patientNames[startIdx:endIdx]))
            for patientName in patientNames[startIdx:endIdx]:
                for sliceId in self.patientSlicesList[patientName]:
                    self.workerInputQueues[workerId].put((patientName, sliceId))
    
    def emptyAllQueues(self):

        try:
            
            for workerId in range(self.numWorkers):
                while not self.workerInputQueues[workerId].empty():
                    try: _ = self.workerInputQueues[workerId].get_nowait()
                    except self.workerInputQueues[workerId].Empty: break

            while not self.workerOutputQueue.empty():
                try: _ = self.workerOutputQueue.get_nowait()
                except self.workerOutputQueue.Empty: break
            
            print (' - [myNewDataloader.emptyAllQueues()] workerInputQueues: {} || workerOutputQueue: {}'.format(
                [self.workerInputQueues[i].qsize() for i in range(self.workerCount)], self.workerOutputQueue.qsize())
            )

        except:
            traceback.print_exc()

    def __iter__(self):
        
        try:
            # Step 0 - Init
            self.fillInputQueues() # once for each epoch
            batchArray, batchMeta = [], []
            
            # Step 1 - Continuously yield results
            while True:
                if not self.workerOutputQueue.empty():

                    # Step 2.1 - Get data point
                    patientSliceArray, patientName, sliceId = self.workerOutputQueue.get(timeout=QUEUE_TIMEOUT)
                    
                    # Step 2.2 - Append to batch
                    if len(batchArray) < self.batchSize:
                        batchArray.append(patientSliceArray)
                        batchMeta.append([patientName, sliceId])
                    
                    # Step 2.3 - Yield batch
                    if len(batchArray) == self.batchSize:
                        batchArray = collate_tensor_fn(batchArray)
                        batchMeta  = np.vstack(batchMeta).T
                        yield batchArray, batchMeta
                        batchArray, batchMeta = [], []
                    
                    # Step 3 - End condition
                    if np.all([self.workerInputQueues[i].empty() for i in range(self.numWorkers)]) and self.workerOutputQueue.empty():
                        break 
        
        except GeneratorExit:
            self.emptyAllQueues()
        
        except KeyboardInterrupt:
            self.closeProcesses()

        except:
            traceback.print_exc()    

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
            for workerId in range(self.numWorkers):
                self.workerInputQueues[workerId].cancel_join_thread() # The cancel_join_thread() method is used to prevent the background thread associated with a queue from joining the main thread when the program exits. By default, when a program terminates, it waits for all non-daemon threads to complete before exiting.
                self.workerInputQueues[workerId].close()
            self.workerOutputQueue.cancel_join_thread()
            self.workerOutputQueue.close()
        finally:
            for workerId in range(self.numWorkers):
                if self.workerProcesses[workerId].is_alive():
                    print (' - [myNewDataloader.closeProcesses()] Worker {} is still alive. Terminating...'.format(workerId))
                    self.workerProcesses[workerId].terminate()        


def getSlice(workerId, inputQueue, outputQueue, stopEvent):
    
    try:
        
        torch.set_num_threads(1)

        while not stopEvent.is_set():
            try:
                patientName, sliceId = inputQueue.get(timeout=QUEUE_TIMEOUT)
                patientSliceArray = getPatientSliceArray(patientName, sliceId)
                outputQueue.put((patientSliceArray, patientName, sliceId))
            except inputQueue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n - [getSlice()][worker={}] function was interrupted.".format(workerId))
    
    except:
        traceback.print_exc()

    finally:
        if stopEvent.is_set():
            print (' - [getSlice] Stopping worker...')

def collate_tensor_fn(batch):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)

##################################################
# main
##################################################

if __name__ == '__main__':
    
    # Step 1 - Setup patient slices (fixed count of slices per patient)
    patientSlicesList = {
        'P1': [45, 67, 32, 21, 69]
        , 'P2': [13, 23, 87, 54, 5]
        , 'P3': [34, 56, 78, 90, 12]
        ,  'P4': [34, 56, 78, 90, 12]
    }
    workerCount, batchSize, epochs = 4, 1, 3
    ################################### Step 2.1 - Create dataset and dataloader
    dataset = myDataset(patientSlicesList, slicesPerPatient=5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # Step 2.2 - Iterate over dataloader
    # print ('\n - [main] Iterating over (my) dataloader...')
    # for epochId in range(epochs):
    #     with tqdm.tqdm(total=len(dataset), desc=' - Epoch {}/{}'.format(epochId+1, epochs)) as pbar:
    #         for i, (patientSliceArray, meta) in enumerate(dataloader):
    #             pbar.update(patientSliceArray.shape[0])

    ################################### Step 3.1 - Create new dataloader
    dataloaderNew = None
    try:
        dataloaderNew = myNewDataloader(patientSlicesList, slicesPerPatient=5, numWorkers=workerCount, batchSize=batchSize)
        print ('\n - [main] Iterating over (myNew) dataloader...')
        for epochId in range(epochs):
            with tqdm.tqdm(total=len(dataset), desc=' - Epoch {}/{}'.format(epochId+1, epochs)) as pbar:
                for i, (X, meta) in enumerate(dataloaderNew):
                    print (' - [main] {}'.format(meta.tolist()))
                    pbar.update(X.shape[0])
        
        dataloaderNew.closeProcesses()
    
    except KeyboardInterrupt:
        if dataloader is not None: dataloader.closeProcesses()

    except:
        traceback.print_exc()
        if dataloaderNew is not None: dataloaderNew.closeProcesses()