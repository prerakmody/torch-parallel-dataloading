
# Import private modules
import src.config as config
import src.utils as utils
import src.dataloader3 as dataloader

# Import public modules
import re
import sys
import pdb
import nrrd
import time
import json
import tqdm
import torch
import monai
import torchio
import traceback
import torchvision
import numpy as np
from pathlib import Path
import torch.multiprocessing as torchMP
# torchMP.set_forkserver_preload(["torch"])

import monai

class Trainer:

    def __init__(self, params):
        
        # Step 0 - Set up the params
        self.params = params

        # Step 1 - Set up other stuff
        self._initData()
        self._initModel()

    def _initData(self):
        
        # Step 1 - Train dataloader
        dataloaderParamsGlobalTrain = self.params[config.KEY_DATALOADER_PARAMS_GLOBAL][config.KEY_TRAIN]
        dirParamsTrain               = dataloaderParamsGlobalTrain[config.KEY_DIR_PARAMS]
        sliceParamsTrain             = dataloaderParamsGlobalTrain[config.KEY_SLICE_PARAMS]
        dataloaderParamsTrain        = dataloaderParamsGlobalTrain[config.KEY_DATALOADER_PARAMS]
        self.trainEpochs             = dataloaderParamsTrain[config.KEY_ANNOTATIONLOADER][config.KEY_EPOCHS]
        self.trainBatchSize          = dataloaderParamsTrain[config.KEY_ANNOTATIONLOADER][config.KEY_BATCH_SIZE]
        self.trainWorkers            = dataloaderParamsTrain[config.KEY_ANNOTATIONLOADER][config.KEY_ANNOTATIONLOADER_WORKERS]
        self.dataloaderTrain         = dataloader.PointAndScribbleDataloader(dirParamsTrain, sliceParamsTrain, dataloaderParamsTrain)
        
        deviceStr = dataloaderParamsTrain[config.KEY_ANNOTATIONLOADER][config.KEY_DEVICE]
        self.device = torch.device(deviceStr)

        # Step 2 - Validation dataloader
        pass

        # Step 3 - Test dataloader
        pass

    def _initModel(self):
        
        # Step 1 - Set up the model
        modelParams = self.params[config.KEY_MODEL_PARAMS]
        neuralNetType = modelParams[config.KEY_NEURALNET]
        modelLR = modelParams[config.KEY_LR]

        self.model = None
        if neuralNetType == config.KEY_UNET_V1:
            self.model = monai.networks.nets.UNet(in_channels=5, out_channels=1, spatial_dims=3, channels=[16, 32, 64, 128], strides=[2, 2, 2]) # [CT,PET, Pred, FGD, Bgd]

        self.model.to(self.device)

        # Step 2 - Set up the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=modelLR)

        # Step 3 - Set up the loss
        lossTypes = modelParams[config.KEY_LOSSES]
        self.losses = []
        for lossType in lossTypes:
            if lossType == config.KEY_LOSS_DICE:
                self.losses.append(monai.losses.DiceLoss(softmax=True))
            elif lossType == config.KEY_LOSS_BCE:
                self.losses.append(monai.losses.FocalLoss(gamma=1, alpha=1))

    def _cleanUp(self):
        self.dataloaderTrain.closeProcesses()

    def train(self):
        
        try:
            
            # Step 0 - Init
            pass

            # Step 1 - Loop over train epochs
            for trainEpoch in range(self.trainEpochs):
                nowStr = utils.getNowStr()
                print ('\n ---------------------------------------------- [epochId={}/{}] ({})'.format(trainEpoch, self.trainEpochs, nowStr))
                
                with tqdm.tqdm(total=len(self.dataloaderTrain)) as pbar:
                    counter = 0
                    
                    # Step 1.1 - Train
                    self.model.train()
                    for (xCT, xPT, yGT, yPred, zFgd, zBgd, meta) in self.dataloaderTrain:

                        pbar.update(self.trainBatchSize)
                        counter += self.trainBatchSize

                        # if counter > 60:
                        #     print (' - [main()] Breaking after 10 iterations')
                        #     break
                    
                    # Step 1.2 - Validate
                    self.model.eval()
            

            # Step 99 - Close
            self._cleanUp()

        except:
            traceback.print_exc()
            pdb.set_trace()

if __name__ == "__main__":

    # Step 0 - Set up directories
    DIR_FILE = Path(__file__).resolve().parent.absolute() # ./src
    DIR_ROOT = DIR_FILE.parent.absolute() # ./
    DIR_DATA = DIR_ROOT / '_data'  
    DIR_MODEL = DIR_ROOT / '_models'
    DIR_TRAINERPARAMS = DIR_FILE / 'trainerParams'

    # Step 1 - Set up the experiment
    # Early Trials
    if 1:
        import src.trainerParams.exp1 as exp

    # Step 2 - Set up the trainer
    trainer = Trainer(exp.params)
    trainer.train()

"""
To-Do
1. Pass through model and get loss
2. Create loss graphs in wandb
"""