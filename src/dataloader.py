# Import private modules
import src.config as config
import src.utils as utils

# Import public modules
import re
import pdb
import nrrd
import time
import tqdm
import torch
import random
# import torchio
import traceback
import torchvision
import numpy as np
from pathlib import Path
import pandas as pd
class PatientDataset(torch.utils.data.Dataset):

    def __init__(self, dirParams, sliceParams, verbose=False):
        
        # Step 1 - Set parameters
        self.dirParams = dirParams
        self.sliceParams = sliceParams
        self.verbose   = verbose

        # Step 2 - Get patient names and paths
        self._getPatientNamesAndPaths()

    def _getPatientNamesAndPaths(self):

        try:
            
            tFiles = time.time()
            dirDataOG = self.dirParams[config.KEY_DIR_DATA_OG]
            fileExt = self.dirParams[config.KEY_EXT]
            regexCT = self.dirParams[config.KEY_REGEX_CT]
            regexPET = self.dirParams[config.KEY_REGEX_PET]
            regexGT = self.dirParams[config.KEY_REGEX_GT]
            regexPred = self.dirParams[config.KEY_REGEX_PRED]

            regexStrFormatCT = str(self.dirParams[config.KEY_STRFMT_CT]).replace('.','\.').replace('{}', '(.+)')

            self.patientPaths = {}
            self.patientNames = {}
            patientId         = 0
            for fileName in Path(dirDataOG).rglob('*' + regexCT + '*'):
                if fileName.suffix == fileExt:
                    match = re.search(regexStrFormatCT, fileName.parts[-1])
                    if match:
                        patientName = match.group(1)    
                        self.patientNames[patientId] = patientName
                        patientId += 1
                        self.patientPaths[patientName] = {
                            config.KEY_CT  : Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_CT].format(patientName),
                            config.KEY_PET : Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_PET].format(patientName),
                            config.KEY_GT  : Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_PRED].format(patientName),
                            config.KEY_PRED: Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_GT].format(patientName)
                        }
            
            assert len(self.patientNames) * 4 == len(list(Path(dirDataOG).glob('*' + fileExt))), 'Number of patients and number of files do not match'
            if self.verbose: print (' - [PatientDataset._getPatientNamesAndPaths()] Time taken: ', round(time.time() - tFiles,2))

        except:
            traceback.print_exc()
            pdb.set_trace()

    def __len__(self):
        return len(self.patientNames)

    def __getitem__(self, idx):
        
        # Step 0 - Init
        patientName = self.patientNames[idx]
        xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal = None, None, None, None, None, None, None, None, None
        badPatientData = True

        # Step 1 - Load patient volumes (and some base preprocessing)
        try:
            
            # Step 1.0 - Debug
            if self.verbose:
                workerId = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0
                # print (' - [PatientDataset.__getitem__()][worker={}] Loading volumes for patient: {}'.format(workerId, patientName))

            # Step 1.1 - Get CT
            arrayCT, _ = nrrd.read(self.patientPaths[patientName][config.KEY_CT])
            if len(arrayCT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] CT volume is not 3D')        

            # Step 1.2 - Get PET
            arrayPT, _ = nrrd.read(self.patientPaths[patientName][config.KEY_PET])
            if len(arrayPT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] PET volume is not 3D')

            # Step 1.3 - Get GT
            arrayMaskGT, _  = nrrd.read(self.patientPaths[patientName][config.KEY_GT])
            if len(arrayMaskGT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] GT volume is not 3D')
            arrayMaskGTLabel = utils.getMaskForLabel(arrayMaskGT, self.sliceParams[config.KEY_LABEL])

            # Step 1.4 - Get Pred
            arrayMaskPred, _ = nrrd.read(self.patientPaths[patientName][config.KEY_PRED])
            if len(arrayMaskPred.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] Pred volume is not 3D')
            arrayMaskPredLabel = utils.getMaskForLabel(arrayMaskPred, self.sliceParams[config.KEY_LABEL])

            # Step 1.5 - Get failure areas
            failureAreasTorch, failureAreasFalseNegativesTorch, failureAreasFalsePositivesTorch = utils.getFailureAreasTorch(arrayMaskGTLabel, arrayMaskPredLabel, self.sliceParams[config.KEY_KSIZE_SEGFAILURE])
            if failureAreasTorch is None: 
                return xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal

            # Step 1.6 - Get failure areas (sorted in descending order of area)
            idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, failureAreaSumAxial, failureAreaSumSagittal, failureAreaSumCoronal = utils.getFailureAreasStatsByView(failureAreasTorch)

            xCT = arrayCT
            xPET = arrayPT
            yGT = arrayMaskGTLabel
            yPred = arrayMaskPredLabel
            errorFP = np.array(failureAreasFalsePositivesTorch)
            errorFN = np.array(failureAreasFalseNegativesTorch)
            badPatientData = False

        except:
            traceback.print_exc()
            pdb.set_trace()
            badPatientData = True
        
        return xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, patientName

class AnnotationDataset(torch.utils.data.Dataset):

    def __init__(self, dirParams, sliceParams, xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, patientName, verbose=False):
        
        # Step 1 - Set parameters
        self.dirParams = dirParams
        self.sliceParams = sliceParams
        self.verbose   = verbose

        # Step 2 - Set patient data
        self.xCT     = np.array(xCT)
        self.xPET    = np.array(xPET)
        self.yGT     = np.array(yGT)
        self.yPred   = np.array(yPred)
        self.errorFP = np.array(errorFP)
        self.errorFN = np.array(errorFN)
        self.idxsSortedAxial    = np.array(idxsSortedAxial)
        self.idxsSortedSagittal = np.array(idxsSortedSagittal)
        self.idxsSortedCoronal  = np.array(idxsSortedCoronal)
        self.patientName = str(patientName)

        # Step 2 - Get patient names and paths
        self.perPatientSlices = 3 * sliceParams[config.KEY_PERVIEW_SLICES]

    def __len__(self):
        return self.perPatientSlices
    
    def __getitem__(self, idx):
        
        # Step 0 - Init
        requestedSliceId     = idx % self.perPatientSlices
        interactionType = self.sliceParams[config.KEY_INTERACTION_TYPE]
        slicesPerView = self.sliceParams[config.KEY_PERVIEW_SLICES]
        axialBool, sagittalBool, coronalBool = False, False, False # for debugging
        z1, z2 = None, None
        
        # Step 1.1 - For axial view
        if 0 <= requestedSliceId < 1*slicesPerView:
        # if 1*slicesPerView <= requestedSliceId < 2*slicesPerView: # [TODO: Just for debugging]
            axialBool = True
            axialIdx  = self.idxsSortedAxial[:slicesPerView][requestedSliceId % 3] 

            interactionDistanceMapVolume, interactionType, interactionClass  = utils.getDistMapVolume(self.errorFP, self.errorFN, axialIdx, config.KEY_AXIAL, self.sliceParams, self.patientName)
            meta = [self.patientName, interactionType, interactionClass, config.KEY_AXIAL, axialIdx]

        # Step 1.2 - For sagittal view
        elif 1*slicesPerView <= requestedSliceId < 2*slicesPerView:
        # elif 0 <= requestedSliceId < 1*slicesPerView: # [TODO: Just for debugging]
        # elif 2*slicesPerView <= requestedSliceId < 3*slicesPerView: # [TODO: Just for debugging]
            sagittalBool = True
            sagittalIdx  = self.idxsSortedSagittal[:slicesPerView][requestedSliceId % 3]

            interactionDistanceMapVolume, interactionType, interactionClass  = utils.getDistMapVolume(self.errorFP, self.errorFN, sagittalIdx, config.KEY_SAGITTAL, self.sliceParams, self.patientName)
            meta = [self.patientName, interactionType, interactionClass, config.KEY_SAGITTAL, sagittalIdx]
        
        # Step 1.3 - For coronal view
        elif 2*slicesPerView <= requestedSliceId < 3*slicesPerView:
        # elif 0 <= requestedSliceId < 1*slicesPerView: # [TODO: Just for debugging]
            coronalBool = True
            coronalIdx  = self.idxsSortedCoronal[:slicesPerView][requestedSliceId % 3]

            interactionDistanceMapVolume, interactionType, interactionClass  = utils.getDistMapVolume(self.errorFP, self.errorFN, coronalIdx, config.KEY_CORONAL, self.sliceParams, self.patientName)
            meta = [self.patientName, interactionType, interactionClass, config.KEY_CORONAL, coronalIdx]

        if interactionDistanceMapVolume is not None:
            if interactionClass == config.KEY_INTERACTION_FGD:
                z1 = torch.tensor(interactionDistanceMapVolume)
                z2 = torch.zeros_like(z1)
            elif interactionClass == config.KEY_INTERACTION_BGD:
                z2 = torch.tensor(interactionDistanceMapVolume)
                z1 = torch.zeros_like(z2)
            
        else:
            viewStr = 'Axial' if axialBool else 'Sagittal' if sagittalBool else 'Coronal'
            print (' - [PointAndScribbleDataset._getPatientData()] interactionDistanceMapVolume is None || requestedPatient: {} || requestedSliceId: {}, view: {}'.format(requestedPatient, requestedSliceId, viewStr))
            z1, z2 = None, None
        
        return z1, z2, meta

class PointAndScribbleDataset(torch.utils.data.Dataset):
    
    def __init__(self, dirParams, sliceParams, transform=None, verbose=False):
        
        # Step 1 - Set parameters
        self.dirParams   = dirParams
        self.sliceParams = sliceParams
        self.transform   = transform
        self.verbose     = verbose

        # Step 2 - Get patient names and paths
        self._getPatientNamesAndPaths()

        # Step 3 - Set current patient object (and other params)
        self.currentPatientObj = {}
        self.perPatientSlices = 3 * sliceParams[config.KEY_PERVIEW_SLICES]

        # Step 4 - Debug ops
        self.loadPatientDataTimes = []

    def _getPatientNamesAndPaths(self):

        try:
            
            dirDataOG = self.dirParams[config.KEY_DIR_DATA_OG]
            fileExt = self.dirParams[config.KEY_EXT]
            regexCT = self.dirParams[config.KEY_REGEX_CT]
            regexPET = self.dirParams[config.KEY_REGEX_PET]
            regexGT = self.dirParams[config.KEY_REGEX_GT]
            regexPred = self.dirParams[config.KEY_REGEX_PRED]

            regexStrFormatCT = str(self.dirParams[config.KEY_STRFMT_CT]).replace('.','\.').replace('{}', '(.+)')

            self.patientPaths = {}
            self.patientNames = {}
            patientId         = 0
            for fileName in Path(dirDataOG).rglob('*' + regexCT + '*'):
                if fileName.suffix == fileExt:
                    match = re.search(regexStrFormatCT, fileName.parts[-1])
                    if match:
                        patientName = match.group(1)    
                        self.patientNames[patientId] = patientName
                        patientId += 1
                        self.patientPaths[patientName] = {
                            config.KEY_CT  : Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_CT].format(patientName),
                            config.KEY_PET : Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_PET].format(patientName),
                            config.KEY_GT  : Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_PRED].format(patientName),
                            config.KEY_PRED: Path(dirDataOG) / self.dirParams[config.KEY_STRFMT_GT].format(patientName)
                        }
            
            assert len(self.patientNames) * 4 == len(list(Path(dirDataOG).glob('*' + fileExt))), 'Number of patients and number of files do not match'

        except:
            traceback.print_exc()
            pdb.set_trace()

    def __len__(self):
        return len(self.patientNames) * self.perPatientSlices

    def _resetCurrentPatientObjToThisPatient(self, requestedPatientName):
         self.currentPatientObj = {requestedPatientName: {
                config.KEY_CT: None, config.KEY_PET: None, config.KEY_GT: None, config.KEY_PRED: None
                , config.KEY_FAILURE_AREAS_FP: None, config.KEY_FAILURE_AREAS_FN: None, config.KEY_SORTED_AXIAL: None, config.KEY_SORTED_SAGITTAL: None, config.KEY_SORTED_CORONAL: None
            }}
         
    def _getPatientData(self, requestedPatient, requestedSliceId):
        """
        Params
        ------
        requestedPatient : str
            Name of the patient.
        requestedSliceId : int
            counter for 3*self.perPatientSlices of the requestedPatient.
        """
        
        x1, x2, y1, y2, z1, z2, meta = None, None, None, None, None, None, None
        badPatientData = False

        try:

            # Step 1 - Load patient volumes (and some base preprocessing)
            tPatientLoad = time.time()
            if self.currentPatientObj[requestedPatient][config.KEY_CT] is None:
                workerId = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0
                # print (' - [PointAndScribbleDataset._getPatientData()][worker={}] Loading volumes for patient: {}'.format(workerId, requestedPatient))
                badPatientData = True
                try:

                    if 0:
                        # Step 1.1 - Get CT
                        if self.currentPatientObj[requestedPatient][config.KEY_CT] is None:
                            arrayCT, _ = nrrd.read(self.patientPaths[requestedPatient][config.KEY_CT])
                            if len(arrayCT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] CT volume is not 3D')
                            badPatientData = True
                            self.currentPatientObj[requestedPatient][config.KEY_CT] = arrayCT

                        # Step 1.2 - Get PET
                        if self.currentPatientObj[requestedPatient][config.KEY_PET] is None:
                            arrayPT, _ = nrrd.read(self.patientPaths[requestedPatient][config.KEY_PET])
                            if len(arrayPT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] PET volume is not 3D')
                            badPatientData = True
                            self.currentPatientObj[requestedPatient][config.KEY_PET] = arrayPT
                        
                        # Step 1.3 - Get GT
                        if self.currentPatientObj[requestedPatient][config.KEY_GT] is None:
                            arrayMaskGT, _  = nrrd.read(self.patientPaths[requestedPatient][config.KEY_GT])
                            if len(arrayMaskGT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] GT volume is not 3D')
                            badPatientData = True
                            arrayMaskGTLabel = utils.getMaskForLabel(arrayMaskGT, self.sliceParams[config.KEY_LABEL])
                            arrayMaskGTLabel = arrayMaskGTLabel.astype(np.uint8)
                            self.currentPatientObj[requestedPatient][config.KEY_GT] = arrayMaskGTLabel

                        # Step 1.4 - Get Pred
                        if self.currentPatientObj[requestedPatient][config.KEY_PRED] is None:
                            arrayMaskPred, _ = nrrd.read(self.patientPaths[requestedPatient][config.KEY_PRED])
                            if len(arrayMaskPred.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] Pred volume is not 3D')
                            badPatientData = True
                            arrayMaskPredLabel = utils.getMaskForLabel(arrayMaskPred, self.sliceParams[config.KEY_LABEL])
                            arrayMaskPredLabel = arrayMaskPredLabel.astype(np.uint8)
                            self.currentPatientObj[requestedPatient][config.KEY_PRED] = arrayMaskPredLabel

                            # Step 1.5 - Get failure areas
                            failureAreasTorch, failureAreasFalseNegativesTorch, failureAreasFalsePositivesTorch = utils.getFailureAreasTorch(arrayMaskGTLabel, arrayMaskPredLabel, self.sliceParams[config.KEY_KSIZE_SEGFAILURE])
                            if failureAreasTorch is None: 
                                return None
                            self.currentPatientObj[requestedPatient][config.KEY_FAILURE_AREAS] = np.array(failureAreasTorch, dtype=np.uint8)
                            self.currentPatientObj[requestedPatient][config.KEY_FAILURE_AREAS_FP] = np.array(failureAreasFalsePositivesTorch, dtype=np.uint8)
                            self.currentPatientObj[requestedPatient][config.KEY_FAILURE_AREAS_FN] = np.array(failureAreasFalseNegativesTorch, dtype=np.uint8)

                            # Step 1.6 - Get failure areas (sorted in descending order of area)
                            idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, failureAreaSumAxial, failureAreaSumSagittal, failureAreaSumCoronal = utils.getFailureAreasStatsByView(failureAreasTorch)
                            self.currentPatientObj[requestedPatient][config.KEY_SORTED_AXIAL] = idxsSortedAxial
                            self.currentPatientObj[requestedPatient][config.KEY_SORTED_SAGITTAL] = idxsSortedSagittal
                            self.currentPatientObj[requestedPatient][config.KEY_SORTED_CORONAL] = idxsSortedCoronal

                            badPatientData = False
                    
                    elif 1:
                        xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal = utils.getPatientData(self.patientPaths[requestedPatient], self.sliceParams, requestedPatient)
                        self.currentPatientObj[requestedPatient][config.KEY_CT] = xCT
                        self.currentPatientObj[requestedPatient][config.KEY_PET] = xPET
                        self.currentPatientObj[requestedPatient][config.KEY_GT] = yGT
                        self.currentPatientObj[requestedPatient][config.KEY_PRED] = yPred
                        self.currentPatientObj[requestedPatient][config.KEY_FAILURE_AREAS_FP] = errorFP
                        self.currentPatientObj[requestedPatient][config.KEY_FAILURE_AREAS_FN] = errorFN
                        self.currentPatientObj[requestedPatient][config.KEY_SORTED_AXIAL] = idxsSortedAxial
                        self.currentPatientObj[requestedPatient][config.KEY_SORTED_SAGITTAL] = idxsSortedSagittal
                        self.currentPatientObj[requestedPatient][config.KEY_SORTED_CORONAL] = idxsSortedCoronal
                        badPatientData = False


                except:
                    traceback.print_exc()
                    pdb.set_trace()
                    badPatientData = True
                
                self.loadPatientDataTimes.append(time.time() - tPatientLoad)
                # if random.random() < 0.1: 
                #     print (' - [PointAndScribbleDataset._getPatientData()] Avg Time taken to load patient data: ', round(np.mean(self.loadPatientDataTimes),2))
                
            # Step 2 - Get volumes (for requestedSliceId) 
            try:
                tAnnotationLoad = time.time()
                if not badPatientData:
                    
                    failureAreasFalsePositives = self.currentPatientObj[requestedPatient][config.KEY_FAILURE_AREAS_FP]
                    failureAreasFalseNegatives = self.currentPatientObj[requestedPatient][config.KEY_FAILURE_AREAS_FN]
                    idxsSortedAxial            = self.currentPatientObj[requestedPatient][config.KEY_SORTED_AXIAL]
                    idxsSortedSagittal         = self.currentPatientObj[requestedPatient][config.KEY_SORTED_SAGITTAL]
                    idxsSortedCoronal          = self.currentPatientObj[requestedPatient][config.KEY_SORTED_CORONAL]

                    if 0:
                        interactionType = self.sliceParams[config.KEY_INTERACTION_TYPE]
                        slicesPerView = self.sliceParams[config.KEY_PERVIEW_SLICES]    
                        axialBool, sagittalBool, coronalBool = False, False, False # for debugging
                        
                        # Step 2.1 - For axial view
                        if 0 <= requestedSliceId < 1*slicesPerView:
                        # if 1*slicesPerView <= requestedSliceId < 2*slicesPerView: # [TODO: Just for debugging]
                            axialBool = True        
                            axialIdx        = idxsSortedAxial[:slicesPerView][requestedSliceId % 3] 
                            interactionDistanceMapVolume, interactionType, interactionClass  = utils.getDistMapVolume(failureAreasFalsePositives, failureAreasFalseNegatives, axialIdx, config.KEY_AXIAL, self.sliceParams, requestedPatient)
                            meta = [requestedPatient, interactionType, interactionClass, config.KEY_AXIAL, axialIdx]

                        # Step 2.2 - For sagittal view
                        elif 1*slicesPerView <= requestedSliceId < 2*slicesPerView:
                        # elif 0 <= requestedSliceId < 1*slicesPerView: # [TODO: Just for debugging]
                        # elif 2*slicesPerView <= requestedSliceId < 3*slicesPerView: # [TODO: Just for debugging]
                            sagittalBool = True
                            sagittalIdx         = idxsSortedSagittal[:slicesPerView][requestedSliceId % 3]
                            interactionDistanceMapVolume, interactionType, interactionClass  = utils.getDistMapVolume(failureAreasFalsePositives, failureAreasFalseNegatives, sagittalIdx, config.KEY_SAGITTAL, self.sliceParams, requestedPatient)
                            meta = [requestedPatient, interactionType, interactionClass, config.KEY_SAGITTAL, sagittalIdx]
                        
                        # Step 2.3 - For coronal view
                        elif 2*slicesPerView <= requestedSliceId < 3*slicesPerView:
                        # elif 0 <= requestedSliceId < 1*slicesPerView: # [TODO: Just for debugging]
                            coronalBool = True
                            coronalIdx         = idxsSortedCoronal[:slicesPerView][requestedSliceId % 3]

                            interactionDistanceMapVolume, interactionType, interactionClass  = utils.getDistMapVolume(failureAreasFalsePositives, failureAreasFalseNegatives, coronalIdx, config.KEY_CORONAL, self.sliceParams, requestedPatient)
                            meta = [requestedPatient, interactionType, interactionClass, config.KEY_CORONAL, coronalIdx]
                        
                    elif 1:
                        meta, interactionClass, interactionDistanceMapVolume = utils.getDistMapVolumeForViews(requestedPatient, requestedSliceId, failureAreasFalsePositives, failureAreasFalseNegatives,
                                                                                            idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, self.sliceParams)
                        
                    if interactionDistanceMapVolume is not None:
                        x1 = torch.tensor(self.currentPatientObj[requestedPatient][config.KEY_CT])
                        x2 = torch.tensor(self.currentPatientObj[requestedPatient][config.KEY_PET])
                        y1 = torch.tensor(self.currentPatientObj[requestedPatient][config.KEY_GT])
                        y2 = torch.tensor(self.currentPatientObj[requestedPatient][config.KEY_PRED])
                        if interactionClass == config.KEY_INTERACTION_FGD:
                            z1 = torch.tensor(interactionDistanceMapVolume)
                            z2 = torch.zeros_like(z1)
                        elif interactionClass == config.KEY_INTERACTION_BGD:
                            z2 = torch.tensor(interactionDistanceMapVolume)
                            z1 = torch.zeros_like(z2)
                        
                        badPatientData = False
                    else:
                        viewStr = 'Axial' if axialBool else 'Sagittal' if sagittalBool else 'Coronal'
                        print (' - [PointAndScribbleDataset._getPatientData()] interactionDistanceMapVolume is None || requestedPatient: {} || requestedSliceId: {}, view: {}'.format(requestedPatient, requestedSliceId, viewStr))
                        x1, x2, y1, y2, z1, z2 = None, None, None, None, None, None     

                    # print (' - [PointAndScribbleDataset._getPatientData()] Time taken to load annotation: ', round(time.time() - tAnnotationLoad,2))         
            
            except:
                traceback.print_exc()
                pdb.set_trace()
                badPatientData = False
                
        except:
            traceback.print_exc()
            pdb.set_trace()
            x1, x2, y1, y2, z1, z2 = None, None, None, None, None, None  

        return x1, x2, y1, y2, z1, z2, meta, badPatientData

    def __getitem__(self, idx):
        
        # Step 1 - Get patient and slice ids
        # print (' - [PointAndScribbleDataset.__getitem__()] idx: ', idx)
        requestedPatientId   = idx // self.perPatientSlices
        requestedSliceId     = idx % self.perPatientSlices
        requestedPatientName = self.patientNames[requestedPatientId]

        # Step 2 - Reset current patient object (if needed)
        if requestedPatientName not in self.currentPatientObj.keys():
            self._resetCurrentPatientObjToThisPatient(requestedPatientName)
        
        # Step 3 - Get patient data
        x1, x2, y1, y2, z1, z2, meta, badPatientData = self._getPatientData(requestedPatientName, requestedSliceId)

        # Step 4 - Apply transformations
        if not badPatientData:
            if self.transform:
                x1 = self.transform(x1)
                x2 = self.transform(x2)
                y1 = self.transform(y1)
                y2 = self.transform(y2)
                z1 = self.transform(z1)
                z2 = self.transform(z2)
        
        if not badPatientData:
            x1 = torch.unsqueeze(x1, -1)
            x2 = torch.unsqueeze(x2, -1)
            y1 = torch.unsqueeze(y1, -1)
            y2 = torch.unsqueeze(y2, -1)
            z1 = torch.unsqueeze(z1, -1)
            z2 = torch.unsqueeze(z2, -1)

        return x1, x2, y1, y2, z1, z2, meta
    
    def show(self, x1, x2, y1, y2, z1, z2, meta, sliceCount=7, contourWidth=0.25):
        """
        Parameters
        ----------
        x1 : torch.Tensor, [B,H,W,D,1]
            CT volume.
        x2 : torch.Tensor, [B,H,W,D,1]
            PET volume.
        y1 : torch.Tensor, [B,H,W,D,1]
            Ground truth volume.
        y2 : torch.Tensor, [B,H,W,D,1]
            Predicted volume.
        z1 : torch.Tensor, [B,H,W,D,1]
            Distance map for interactionClass=foreground interaction (interactionType=[scribles,points])
        z2 : torch.Tensor, [B,H,W,D,1]
            Distance map for interactionClass=background interaction (interactionType=[scribles,points])
        meta : list
            [patientName, interactionType, interactionClass, view, sliceId]
        """

        try:
            
            # Step 0 - Init
            batchSize = x1.shape[0]
            import matplotlib.pyplot as plt
            # reduce hspace and wspace
            f,axarr = plt.subplots(batchSize, 2+sliceCount, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10,4))
            if batchSize == 1: axarr = [axarr]
            f.subplots_adjust(hspace=0.1, wspace=0.1)
            cmapPet = utils.convertColorMapWithLinearTransparency(plt.cm.Oranges)
            cmapDistMapFgd = utils.convertColopMapWithZeroAsTransparent(plt.cm.Blues)
            cmapDistMapBgd = utils.convertColopMapWithZeroAsTransparent(utils.getYellowColorMap())

            def checkDistMapSanity(distMapSlice, isInteractionSlice, metaData):
                
                error = False
                if isInteractionSlice:
                    if np.max(distMapSlice) != 1.0:
                        error = True
                else:
                    if np.max(distMapSlice) == 1.0:
                        error = True
                
                if error:
                    print ('   -- [PointAndScribbleDataset.show().checkDistMapSanity()] Something wrong with distance map: ', metaData)
            
            for batchId in range(batchSize):
                
                # Step 1 - Get meta
                patientName      = meta[0][batchId]
                interactionType  = meta[1][batchId]
                interactionClass = meta[2][batchId]
                view             = meta[3][batchId]
                sliceId          = meta[4][batchId]
                metaDataStr = ' - {} - {} - {} - {} - {}'.format(patientName, interactionType, interactionClass, view, sliceId)

                # Step 2 - Get slices
                if view == config.KEY_AXIAL:
                    slicesCT = x1[batchId, :, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, 0].detach().cpu().numpy()
                    slicesPT = x2[batchId, :, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, 0].detach().cpu().numpy()
                    slicesGT = y1[batchId, :, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, 0].detach().cpu().numpy()
                    slicesPred = y2[batchId, :, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, 0].detach().cpu().numpy()
                    slicesZ1 = z1[batchId, :, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, 0].detach().cpu().numpy() # [H,W, sliceCount]
                    slicesZ2 = z2[batchId, :, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, 0].detach().cpu().numpy() # [H,W, sliceCount]
                elif view == config.KEY_SAGITTAL:
                    slicesCT = x1[batchId, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, :, 0].detach().cpu().numpy()
                    slicesPT = x2[batchId, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, :, 0].detach().cpu().numpy()
                    slicesGT = y1[batchId, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, :, 0].detach().cpu().numpy()
                    slicesPred = y2[batchId, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, :, 0].detach().cpu().numpy()
                    slicesZ1 = z1[batchId, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, :, 0].detach().cpu().numpy() # [sliceCount,H,W]
                    slicesZ2 = z2[batchId, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, :, 0].detach().cpu().numpy() 
                elif view == config.KEY_CORONAL:
                    slicesCT = x1[batchId, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, 0].detach().cpu().numpy()
                    slicesPT = x2[batchId, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, 0].detach().cpu().numpy()
                    slicesGT = y1[batchId, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, 0].detach().cpu().numpy()
                    slicesPred = y2[batchId, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, 0].detach().cpu().numpy()
                    slicesZ1 = z1[batchId, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, 0].detach().cpu().numpy() # [H,sliceCount,W]
                    slicesZ2 = z2[batchId, :, sliceId-sliceCount//2: sliceId+sliceCount//2+1, :, 0].detach().cpu().numpy()

                # Step 3 - Plot
                    
                # Step 3.1 - First two columns
                axarr[batchId, 0].set_ylabel('Patient: {} \n Type: {} \n Class: {} \n View: {}'.format(patientName, interactionType, interactionClass, view))
                if view == config.KEY_AXIAL:
                    axarr[batchId, 0].imshow(slicesCT[:,:,sliceCount//2], cmap='gray')
                    axarr[batchId, 0].imshow(slicesPT[:,:,sliceCount//2], cmap=cmapPet)
                    axarr[batchId, 1].imshow(slicesCT[:,:,sliceCount//2], cmap='gray')
                    axarr[batchId, 1].imshow(slicesPT[:,:,sliceCount//2], cmap=cmapPet)
                    axarr[batchId, 1].contour(slicesGT[:,:,sliceCount//2], colors=config.COLOR_GREEN, linewidths=contourWidth)
                    axarr[batchId, 1].contour(slicesPred[:,:,sliceCount//2], colors=config.COLOR_RED, linewidths=contourWidth, linestyles='dotted')
                elif view == config.KEY_SAGITTAL:
                    axarr[batchId, 0].imshow(slicesCT[sliceCount//2,:,:], cmap='gray')
                    axarr[batchId, 0].imshow(slicesPT[sliceCount//2,:,:], cmap=cmapPet)
                    axarr[batchId, 1].imshow(slicesCT[sliceCount//2,:,:], cmap='gray')
                    axarr[batchId, 1].imshow(slicesPT[sliceCount//2,:,:], cmap=cmapPet)
                    axarr[batchId, 1].contour(slicesGT[sliceCount//2,:,:], colors=config.COLOR_GREEN, linewidths=contourWidth)
                    axarr[batchId, 1].contour(slicesPred[sliceCount//2,:,:], colors=config.COLOR_RED, linewidths=contourWidth, linestyles='dotted')
                elif view == config.KEY_CORONAL:
                    axarr[batchId, 0].imshow(slicesCT[:,sliceCount//2,:], cmap='gray')
                    axarr[batchId, 0].imshow(slicesPT[:,sliceCount//2,:], cmap=cmapPet)
                    axarr[batchId, 1].imshow(slicesCT[:,sliceCount//2,:], cmap='gray')
                    axarr[batchId, 1].imshow(slicesPT[:,sliceCount//2,:], cmap=cmapPet)
                    axarr[batchId, 1].contour(slicesGT[:,sliceCount//2,:], colors=config.COLOR_GREEN, linewidths=contourWidth)
                    axarr[batchId, 1].contour(slicesPred[:,sliceCount//2,:], colors=config.COLOR_RED, linewidths=contourWidth, linestyles='dotted')

                # Step 3.2 - Remaining columns
                for sliceCount_ in range(sliceCount):
                    
                    titleStr = 'Slice: {}'.format(sliceId - sliceCount//2 + sliceCount_)
                    if sliceCount_ == sliceCount//2:
                        titleStr += ' **'

                    if view == config.KEY_AXIAL:
                        axarr[batchId, sliceCount_ + 2].imshow(slicesCT[:,:,sliceCount_], cmap='gray')
                        axarr[batchId, sliceCount_ + 2].contour(slicesGT[:,:,sliceCount_], colors=config.COLOR_GREEN, linewidths=contourWidth)
                        axarr[batchId, sliceCount_ + 2].contour(slicesPred[:,:,sliceCount_], colors=config.COLOR_RED, linewidths=contourWidth, linestyles='dotted')
                        axarr[batchId, sliceCount_ + 2].imshow(slicesZ1[:,:,sliceCount_], cmap=cmapDistMapFgd, vmin=0, vmax=1)
                        axarr[batchId, sliceCount_ + 2].imshow(slicesZ2[:,:,sliceCount_], cmap=cmapDistMapBgd, vmin=0, vmax=1)
                        if sliceCount_ == sliceCount//2:
                            if interactionClass == config.KEY_INTERACTION_FGD: checkDistMapSanity(slicesZ1[:,:,sliceCount_], True, metaDataStr)
                            elif interactionClass == config.KEY_INTERACTION_BGD: checkDistMapSanity(slicesZ2[:,:,sliceCount_], True, metaDataStr)
                        else:
                            if interactionClass == config.KEY_INTERACTION_FGD: checkDistMapSanity(slicesZ1[:,:,sliceCount_], False, metaDataStr)
                            elif interactionClass == config.KEY_INTERACTION_BGD: checkDistMapSanity(slicesZ2[:,:,sliceCount_], False, metaDataStr)

                    elif view == config.KEY_SAGITTAL:
                        axarr[batchId, sliceCount_ + 2].imshow(slicesCT[sliceCount_,:,:], cmap='gray')
                        axarr[batchId, sliceCount_ + 2].contour(slicesGT[sliceCount_,:,:], colors=config.COLOR_GREEN, linewidths=contourWidth)
                        axarr[batchId, sliceCount_ + 2].contour(slicesPred[sliceCount_,:,:], colors=config.COLOR_RED, linewidths=contourWidth, linestyles='dotted')
                        axarr[batchId, sliceCount_ + 2].imshow(slicesZ1[sliceCount_, :, :], cmap=cmapDistMapFgd, vmin=0, vmax=1)
                        axarr[batchId, sliceCount_ + 2].imshow(slicesZ2[sliceCount_, :, :], cmap=cmapDistMapBgd, vmin=0, vmax=1)
                    
                    elif view == config.KEY_CORONAL:
                        axarr[batchId, sliceCount_ + 2].imshow(slicesCT[:,sliceCount_,:], cmap='gray')
                        axarr[batchId, sliceCount_ + 2].contour(slicesGT[:,sliceCount_,:], colors=config.COLOR_GREEN, linewidths=contourWidth)
                        axarr[batchId, sliceCount_ + 2].contour(slicesPred[:,sliceCount_,:], colors=config.COLOR_RED, linewidths=contourWidth, linestyles='dotted') 
                        axarr[batchId, sliceCount_ + 2].imshow(slicesZ1[:, sliceCount_, :], cmap=cmapDistMapFgd, vmin=0, vmax=1)
                        axarr[batchId, sliceCount_ + 2].imshow(slicesZ2[:, sliceCount_, :], cmap=cmapDistMapBgd, vmin=0, vmax=1)
                        
                    
                    axarr[batchId, sliceCount_ + 2].set_title(titleStr)
            
            plt.show(block=False)
            pdb.set_trace()

        except:
            traceback.print_exc()
            pdb.set_trace()

if __name__ == "__main__":
    
    dt_all = {'workers': [],
                        'batch_size': [],
                        'cost_time': [],
                        'speed': [],
                        }
    for workers_nb in [1,2,4,8]:
        for bs in [1,2,4,8]:
            print(f"workers: {workers_nb}, batch size: {bs} ---------start--------")
                    
            DIR_FILE = Path(__file__).resolve().parent.absolute() # ./src
            DIR_ROOT = DIR_FILE.parent.absolute() # ./
            DIR_DATA = DIR_ROOT / '_data'    

            # Step 1 - Dataloader params
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
                
                # Define transformations
                transform = torchvision.transforms.Compose([
                #     transforms.Resize((256, 256)),
                #     transforms.ToTensor()
                ])
            
            # just PointAndScribbleDataset (baseline)
            if 1:
                # Step 2 - Create dataset
                dataset = PointAndScribbleDataset(dirParams, sliceParams, transform=transform)
                
                # Step 3 - Create dataloader
                epochs = 4
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False
                                                        , num_workers=workers_nb #, prefetch_factor=2
                                                        , pin_memory=True, pin_memory_device=config.nameGPU
                                                        # , num_workers=0, pin_memory=True
                                                        )
                
                # Iterate over the dataset
                for epoch_ in range(epochs):
                    print (' ------------------------------------- Epoch: ', epoch_)
                    t1 = time.time()
                    with tqdm.tqdm(total=len(dataset), desc='Epoch: {}'.format(epoch_)) as pbar:
                        for i, (x1,x2,y1,y2,z1,z2, meta) in enumerate(dataloader):
                            x1 = x1.to(config.nameGPU)
                            x2 = x2.to(config.nameGPU)
                            y1 = y1.to(config.nameGPU)
                            y2 = y2.to(config.nameGPU)
                            z1 = z1.to(config.nameGPU)
                            z2 = z2.to(config.nameGPU)
                            # print (' -------------- meta: ', meta)
                            # dataset.show(x1, x2, y1, y2, z1, z2, meta, sliceCount=7)
                            
                            # if i == 5:
                            #     break
                            # pdb.set_trace()
                            pbar.update(x1.shape[0])
                    t2 = time.time()
                    dt = t2-t1 
                    print(f"time cost: {dt}, speed: {855/dt}")
                    dt_all['workers'].append(workers_nb)
                    dt_all['batch_size'].append(bs)
                    dt_all['cost_time'].append(dt)
                    dt_all['speed'].append(855/dt)
          
            # just PatientDataset
            elif 0:

                t0 = time.time()
                epochs = 3
                patientDataset = PatientDataset(dirParams, sliceParams, verbose=False)
                patientDataloader = torch.utils.data.DataLoader(patientDataset, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=2, persistent_workers=True)
                for _ in range(epochs):
                    with tqdm.tqdm(total=len(patientDataset), desc='Epoch: {}'.format(0)) as pbar:
                        for i, (xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal) in enumerate(patientDataloader):
                            # print (' - [main] i: ', i, xCT.shape)
                            pbar.update(xCT.shape[0])
                print (' - [main] Time taken: ', round(time.time() - t0),2, 's')
            
            # patientDataset and annotationDataset (v1)
            elif 0:

                t0 = time.time()
                epochs = 3
                patientDataset = PatientDataset(dirParams, sliceParams, verbose=False)
                patientDataloader = torch.utils.data.DataLoader(patientDataset, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=1, persistent_workers=True)
                for _ in range(epochs):
                    with tqdm.tqdm(total=len(patientDataset) * sliceParams[config.KEY_PERVIEW_SLICES] * 3, desc=' - Epoch: {}'.format(0)) as pbar:
                        for i, (xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, patientName) in enumerate(patientDataloader):
                            print (' - [main] i: ', i, patientName)
                            annotationDataset = AnnotationDataset(dirParams, sliceParams, xCT[0], xPET[0], yGT[0], yPred[0], errorFP[0], errorFN[0], idxsSortedAxial[0], idxsSortedSagittal[0], idxsSortedCoronal[0], patientName[0], verbose=False)
                            annotationDataloader = torch.utils.data.DataLoader(annotationDataset, batch_size=4, shuffle=False, num_workers=4, prefetch_factor=1)
                            for (z1, z2, meta) in annotationDataloader:
                                print (' - [main] patientName: {} || meta: {}'.format(patientName, meta))
                                pbar.update(4)
            
            # patientDataset and annotationDataset (v2)
            elif 0:

                t0 = time.time()
                epochs = 3
                patientDataset = PatientDataset(dirParams, sliceParams, verbose=False)
                patientDataloader = torch.utils.data.DataLoader(patientDataset, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=1, persistent_workers=True)
                for _ in range(epochs):
                    with tqdm.tqdm(total=len(patientDataset) * sliceParams[config.KEY_PERVIEW_SLICES] * 3, desc=' - Epoch: {}'.format(0)) as pbar:
                        for i, (xCT, xPET, yGT, yPred, errorFP, errorFN, idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, patientName) in enumerate(patientDataloader):
                            print ('\n - [main] i: ', i, patientName)

                            if 0:
                                if i == 0:
                                    annotationDataset = AnnotationDataset(dirParams, sliceParams, xCT[0], xPET[0], yGT[0], yPred[0], errorFP[0], errorFN[0], idxsSortedAxial[0], idxsSortedSagittal[0], idxsSortedCoronal[0], patientName[0], verbose=False)
                                    annotationDataloader = torch.utils.data.DataLoader(annotationDataset, batch_size=4, shuffle=False, num_workers=4, prefetch_factor=4)
                                else:
                                    annotationDataset.xCT = np.array(xCT[0])
                                    annotationDataset.xPET = np.array(xPET[0])
                                    annotationDataset.yGT = np.array(yGT[0])
                                    annotationDataset.yPred = np.array(yPred[0])
                                    annotationDataset.errorFP = np.array(errorFP[0])
                                    annotationDataset.errorFN = np.array(errorFN[0])
                                    annotationDataset.idxsSortedAxial = np.array(idxsSortedAxial[0])
                                    annotationDataset.idxsSortedSagittal = np.array(idxsSortedSagittal[0])
                                    annotationDataset.idxsSortedCoronal = np.array(idxsSortedCoronal[0])
                                    annotationDataset.patientName = patientName[0]
                                
                            for (z1, z2, meta) in annotationDataloader:
                                print (' - [main] patientName: {} || meta: {}'.format(patientName, meta))
                                pbar.update(4)

                print (' - [main] Time taken: ', round(time.time() - t0),2, 's')
        
    df = pd.DataFrame(dt_all)
    df.to_csv('dataloader1_table.csv', index=False)  # index=False 表示不保存索引
    print(f"finishe all!")
        """
        Data: Z:\2021_HECKTOR_HNTumorAuto\_models\FocusNetV4ResV2-LR10e3I20__ValCenterFold5-B2-NewWindowing__CEF095B005__seed42\ckpt_epoch1000\images\Test\patches
        """

        """
        I have a pytorch dataset that reads 3D volumes, does some heavy processing, and retuns them. 
        I need to create an array of max length=N, that is shared between the workers. 
        So if a patients volume is already read by a worker, the other workers simply access that in shared memory. 

        These volumes are then read by another dataset that turns these volumes into slices, again with some heavy processing. 

        How do I build these dataset and dataloaders using torch.multiprocessing to do this?
        """