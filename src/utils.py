# Import private libraries
import src.config as config

# Import public libraries
import re
import sys
import pdb
import time
import nrrd
import torch
import psutil
import datetime
import traceback
import numpy as np
import scipy.ndimage
import skimage.measure
from pathlib import Path
import skimage.morphology
import matplotlib.pyplot as plt

####################################################
# FILE I/O AND SYSTEM
####################################################

def readNrrdFile(file):

    data, header = None, None

    if Path(file).exists():
        if Path(file).suffix == config.EXT_NRRD:
            data, header = nrrd.read(file)
        else:
            print (' - [utils.readNrrdFile] File is not a nrrd file: ' + file)
    else:
        print (' - [utils.readNrrdFile] File does not exist: ' + file)
    
    return data, header

def getVirtualCPUCount():

    virtualCPUCount = -1

    try:
        
        if sys.platform == 'linux':
            with open('/proc/cpuinfo') as f:
                data = f.read()
                # m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', open('/proc/self/status').read())
                m = re.search(r'(?m)^processor\s*:\s*(.*)$', data)
                if m:
                    virtualCPUCount = bin(int(m.group(1).replace(',', ''), 16)).count('1')
        elif sys.platform == 'win32':
            virtualCPUCount = psutil.cpu_count(logical=True)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return virtualCPUCount

def getPerCPUThreadCount():
    
        perCPUThreadCount = -1
    
        try:
            
            perCPUThreadCount = psutil.cpu_count(logical=True) / psutil.cpu_count(logical=False)
    
        except:
            traceback.print_exc()
            pdb.set_trace()
        
        return int(perCPUThreadCount)

def getNowStr():
        
    nowStr = None
    
    try:
        
        nowStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return nowStr

####################################################
# DIST MAP OPERATIONS
####################################################

def getDistanceMapOld(binaryMask, distType, **kwargs):
    """
    Params
    ------
    binaryMask: torch.Tensor, [H,W,depth], containing 1s and 0s
    distType: str, config.KEY_DIST_EUCLIDEAN, config.KEY_DIST_GEODESIC, config.KEY_DIST_GAUSSIAN
    """

    # Step 1 - Compute Euclidean distance
    # euclideanDistanceMap = torch.cdist(binaryMask, binaryMask, p=2) # No, didnt work
    # euclideanDistanceMap = torch.cdist(binaryMask, 1-binaryMask, p=2) # No, didnt work
    # euclideanDistanceMap = torch.cdist(1-binaryMask, 1-binaryMask, p=2) # No, didnt work
    # euclideanDistanceMap = torch.cdist(1-binaryMask, binaryMask, p=2) # No, didnt work
    euclideanDistanceMap = torch.cdist(binaryMask, binaryMask, p=1) 
    
    # Step 2 - if gaussian, apply function
    if distType == config.KEY_DIST_GAUSSIAN:
        distanceMap = torch.exp(-euclideanDistanceMap**2 / (2 * kwargs[config.KEY_GAUSSIAN_SIGMA]**2))
    elif distType == config.KEY_DIST_GEODESIC:
        pass
    elif distType == config.KEY_DIST_EUCLIDEAN:
        distanceMap = euclideanDistanceMap
    else:
        distanceMap = euclideanDistanceMap
    
    # Spte 3 - Normalize
    distanceMap = distanceMap / distanceMap.max()
    
    return distanceMap

def getGaussianDistanceMap(binaryMask, distZ, sigma, view, patientId, sliceId, show=False, sliceCount=None):
    """
    Params
    ------
    binaryMask: torch.Tensor, [H,W,depth], containing 1s and 0s
    sliceId: int, slice id
    distZ: int, distance in z direction
    sigma: float, sigma for gaussian
    show: bool, whether to show the distance map
    sliceCount: int, number of slices to show, # keep it odd

    - Note: keeping sigma same, low distZ means more slices | keeping distZ same, low sigma less slices
    """

    gaussianDistanceMap = None
    try:
        
        # Step 0 - init
        if np.sum(binaryMask) == 0:
            print (' - [getGaussianDistanceMap()] binaryMask is empty for patientId=' + patientId + ', sliceId=' + str(sliceId) + ', view=' + view)

        # Step 1 - Get euclidean distance map
        euclideanDistanceMap = scipy.ndimage.distance_transform_edt(1-binaryMask, sampling=(1,1,distZ))
        maxVal = euclideanDistanceMap.max()
        euclideanDistanceMap = 1 - (euclideanDistanceMap / maxVal)
        
        # Step 2 - Get gaussian distance map
        gaussianDistanceMap = np.exp(-(1-euclideanDistanceMap)**2 / (2 * sigma**2))

        # Step 99 - Show
        if show and sliceCount is not None:
            f,axarr = plt.subplots(3,sliceCount);
            for showId_, sliceId_ in enumerate(range(sliceId-sliceCount//2, sliceId+sliceCount//2 + 1)):
                axarr[0][showId_].set_title('Slice ' + str(sliceId_));
                if view == config.KEY_AXIAL:
                    axarr[0][showId_].imshow(binaryMask[:,:,sliceId_]);
                    axarr[1][showId_].imshow(euclideanDistanceMap[:,:,sliceId_], vmin=0, vmax=1, cmap='Oranges');
                    axarr[2][showId_].imshow(gaussianDistanceMap[:,:,sliceId_], vmin=0, vmax=1, cmap='Oranges');
                    axarr[2][showId_].set_title(round(np.sum(gaussianDistanceMap[:,:,sliceId_]),1))
                elif view == config.KEY_SAGITTAL:
                    axarr[0][showId_].imshow(binaryMask[sliceId_,:,:]);
                    axarr[1][showId_].imshow(euclideanDistanceMap[sliceId_,:,:], vmin=0, vmax=1, cmap='Oranges');
                    axarr[2][showId_].imshow(gaussianDistanceMap[sliceId_,:,:], vmin=0, vmax=1, cmap='Oranges');
                    axarr[2][showId_].set_title(round(np.sum(gaussianDistanceMap[sliceId_,:,:]),1))
                elif view == config.KEY_CORONAL:
                    axarr[0][showId_].imshow(binaryMask[:,sliceId_,:]);
                    axarr[1][showId_].imshow(euclideanDistanceMap[:,sliceId_,:], vmin=0, vmax=1, cmap='Oranges');
                    axarr[2][showId_].imshow(gaussianDistanceMap[:,sliceId_,:], vmin=0, vmax=1, cmap='Oranges');
                    axarr[2][showId_].set_title(round(np.sum(gaussianDistanceMap[:,sliceId_,:]),1))

            plt.suptitle(
                'PatientId=' + patientId + ', sliceId=' + str(sliceId) + ', view=' + view + '\n' +
                'Mask Shape=' + str(binaryMask.shape) + '\n sigma=' + str(sigma) + ', distZ=' + str(distZ))
            f.show() # so that it does not interfere with any superseding plt.show() calls
            # pdb.set_trace()    
    
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return gaussianDistanceMap

def getBezierCurve(t, points):
    """
    Params
    ------
    t: range
    points: [N,2]
    """

    curve = None
    
    try:

        def getBezierCurvePointsMeh(t, points):
            B = (1 - t)**2 * points[0] + 2 * (1 - t) * t * points[1] + t**2 * points[2]
            return B

        def getBezierCurvePoints(t, points):
            import scipy.special
            n = len(points)
            B = np.zeros(2)
            for i in range(n): B += scipy.special.comb(n-1, i) * ((1-t)**(n-1-i)) * (t**i) * points[i]
            return B
    
        curve = np.array([getBezierCurvePoints(ti, points) for ti in t])

    except:
        traceback.print_exc()
        pdb.set_trace()

    return curve

def getRandomBezierCurveOnMask(binaryMask, slice, view, numPoints=3):
    """
    Params
    ------
    binaryMask: torch.Tensor, [H,W,depth], containing only 0s
    numPoints: int, number of points to get for the bezier curve
    """

    curve = None
    
    try:

        # Step 1 - Get random points
        binaryMaskXLen, binaryMaskYLen, binaryMaskZLen = binaryMask.shape[0], binaryMask.shape[1], binaryMask.shape[2]
        randomPointsX  = np.random.randint(0, binaryMaskXLen, numPoints)
        randomPointsY  = np.random.randint(0, binaryMaskYLen, numPoints)
        randomPointsZ  = np.random.randint(0, binaryMaskZLen, numPoints)

        if view == config.KEY_AXIAL:
            randomPoints2D = np.array([randomPointsX, randomPointsY]).T # [numPoints, 2]
            # print (randomPoints2D)
            randomPoints2D = np.array([[117,115], [111,32], [42,48]])
        elif view == config.KEY_SAGITTAL:
            randomPoints2D = np.array([randomPointsY, randomPointsZ]).T
            # print (view, randomPoints2D)
            randomPoints2D = np.array([[95,75], [21,92], [127,47]])
        elif view == config.KEY_CORONAL:
            randomPoints2D = np.array([randomPointsX, randomPointsZ]).T

        if 0:
            print (view, randomPoints2D)
            # randomPoints = np.array([[25,12], [5,44], [8,17]])  # small mustachy stroke
            # randomPoints = np.array([[52,1], [10,56], [3,47]])  # big stroke
            # randomPoints = np.array([[17,43], [41,14], [28,11]])  # nike swoosh
        
        # Step 2 - Get the bezier curve
        t = np.linspace(0, 1, 100)
        curve = getBezierCurve(t, randomPoints2D)

        # Step 3 - Set the curve to 1 in the binaryMask
        curveInt = curve.astype(int)
        for i in range(curveInt.shape[0]):
            if view == config.KEY_AXIAL:
                binaryMask[curveInt[i,0], curveInt[i,1], slice] = 1
            elif view == config.KEY_SAGITTAL:
                binaryMask[slice, curveInt[i,0], curveInt[i,1]] = 1
                # binaryMask[slice, curveInt[i,1], curveInt[i,0]] = 1 # simply rotates it
            elif view == config.KEY_CORONAL:
                binaryMask[curveInt[i,0], slice, curveInt[i,1]] = 1

    except:
        traceback.print_exc()
        pdb.set_trace()

    return binaryMask

def getMedialAxisPointsForBinarySlice(binaryErrorSlice, sliceId, view):
    """
    Params
    ------
    binaryErrorSlice: [H,W], np.ndarray, containing 1s and 0s
    sliceId: int, slice id
    view: str, config.KEY_AXIAL, config.KEY_SAGITTAL, config.KEY_CORONAL
    """

    medialAxisPointsForVolume = None
    
    try:
        
        # Step 0 - Find connected components and select a random one
        binaryErrorSliceWithComponents, componentCount = skimage.measure.label(binaryErrorSlice, return_num=True)
        if componentCount > 1:
            randomComponent = np.random.randint(1, componentCount+1)
            binaryErrorSliceSingleComponent = getMaskForLabel(binaryErrorSliceWithComponents, randomComponent)
            # f,axarr = plt.subplots(1,3); axarr[0].imshow(binaryErrorSlice); axarr[1].imshow(binaryErrorSliceWithComponents); axarr[2].imshow(binaryErrorSliceSingleComponent); f.show(); pdb.set_trace()
        else:
            binaryErrorSliceSingleComponent = binaryErrorSlice

        # Step 1 - Get medial axis
        skeletonizedMaskSlice = skimage.morphology.medial_axis(np.ascontiguousarray(binaryErrorSliceSingleComponent))
        # plt.imshow(binaryErrorSlice); plt.imshow(skeletonizedMaskSlice, alpha=0.5); plt.show(block=False)
        # pdb.set_trace()

        # Step 2.1 - Get medial axis points
        medialAxisPointsForSlice = np.argwhere(skeletonizedMaskSlice) # these are 2D points
        medialAxisPointsForSlice = medialAxisPointsForSlice.astype(int)
        # plt.imshow(binaryErrorSlice); plt.scatter(medialAxisPointsForSlice[:,1], medialAxisPointsForSlice[:,0], c='r', s=1); plt.show()
        
        # Step 3.2  - Convert to 3D points
        if view == config.KEY_AXIAL:
            medialAxisPointsForVolume = np.zeros((medialAxisPointsForSlice.shape[0], 3))
            medialAxisPointsForVolume[:,0] = medialAxisPointsForSlice[:,0]
            medialAxisPointsForVolume[:,1] = medialAxisPointsForSlice[:,1]
            medialAxisPointsForVolume[:,2] = sliceId
            medialAxisPointsForVolume = medialAxisPointsForVolume.astype(int)
        
        elif view == config.KEY_SAGITTAL:
            medialAxisPointsForVolume = np.zeros((medialAxisPointsForSlice.shape[0], 3))
            medialAxisPointsForVolume[:,0] = medialAxisPointsForSlice[:,0]
            medialAxisPointsForVolume[:,1] = medialAxisPointsForSlice[:,1]
            medialAxisPointsForVolume[:,2] = sliceId
            medialAxisPointsForVolume = medialAxisPointsForVolume.astype(int)
        
        elif view == config.KEY_CORONAL:
            medialAxisPointsForVolume = np.zeros((medialAxisPointsForSlice.shape[0], 3))
            medialAxisPointsForVolume[:,0] = medialAxisPointsForSlice[:,0]
            medialAxisPointsForVolume[:,1] = medialAxisPointsForSlice[:,1]
            medialAxisPointsForVolume[:,2] = sliceId
            medialAxisPointsForVolume = medialAxisPointsForVolume.astype(int)

    except:
        traceback.print_exc()
        pdb.set_trace()

    return medialAxisPointsForVolume

def getDistMapVolume(volumeFP, volumeFN, sliceId, view, interactionParams, patientId):
    """
    Params
    ------
    volumeFP: [H,W,Depth], np.ndarray, containing 1s and 0s
    volumeFN: [H,W,Depth], np.ndarray, containing 1s and 0s
    sliceId: int, slice id
    view: str, config.KEY_AXIAL, config.KEY_SAGITTAL, config.KEY_CORONAL
    interactionParams: dict, containing interaction parameters
        - KEY_INTERACTION_TYPE: list[str], config.KEY_INTERACTION_POINTS, config.KEY_INTERACTION_SCRIBBLES
    patientId: str, patient id
    """

    interactionDistanceMapVolume = None
    interactionType  = np.random.choice(interactionParams[config.KEY_INTERACTION_TYPE])
    interactionClass = None

    try:

        # Step 1 - Get slice
        if view == config.KEY_AXIAL:
            sliceFP = volumeFP[:,:,sliceId]
            sliceFN = volumeFN[:,:,sliceId]
        elif view == config.KEY_SAGITTAL:
            sliceFP = volumeFP[sliceId,:,:]
            sliceFN = volumeFN[sliceId,:,:]
        elif view == config.KEY_CORONAL:
            sliceFP = volumeFP[:,sliceId,:]
            sliceFN = volumeFN[:,sliceId,:]
        else:
            print (' - [getScribble()] Invalid view: ' + view)
            return interactionDistanceMapVolume, interactionType, interactionClass

        # Step 2 - Get an error mask
        chosenSliceMask = None
        if np.sum(sliceFP) and np.sum(sliceFN):
            if np.random.rand() > 0.5:
                chosenSliceMask = sliceFP
                interactionClass = config.KEY_INTERACTION_BGD
            else:
                chosenSliceMask = sliceFN
                interactionClass = config.KEY_INTERACTION_FGD
        elif np.sum(sliceFP):
            chosenSliceMask = sliceFP
            interactionClass = config.KEY_INTERACTION_BGD
        elif np.sum(sliceFN):
            chosenSliceMask = sliceFN
            interactionClass = config.KEY_INTERACTION_FGD
        else:
            print (' - [getScribble()] No FP or FN found')
            return interactionDistanceMapVolume, interactionType, interactionClass
        
        # Step 3 - Get 3D coords of interaction points (regular points or scribbles)
        if interactionType == config.KEY_INTERACTION_POINTS:
            pass
        elif interactionType == config.KEY_INTERACTION_SCRIBBLES:
            scribbleType = np.random.choice(interactionParams[config.KEY_SCRIBBLE_TYPE])     
            if scribbleType == config.KEY_SCRIBBLE_RANDOM:
                pass
            elif scribbleType == config.KEY_SCRIBBLE_MEDIAL_AXIS:
                interactionPointsIn3D = getMedialAxisPointsForBinarySlice(chosenSliceMask, sliceId, view)
        
        # Step 4 - Apply 3D scribble points to the volume
        interactionPointsVolume = np.zeros(volumeFP.shape)
        if view == config.KEY_AXIAL:
            interactionPointsVolume[interactionPointsIn3D[:,0], interactionPointsIn3D[:,1], sliceId] = 1
        elif view == config.KEY_SAGITTAL:
            interactionPointsVolume[sliceId, interactionPointsIn3D[:,0], interactionPointsIn3D[:,1]] = 1
        elif view == config.KEY_CORONAL:
            interactionPointsVolume[interactionPointsIn3D[:,0], sliceId, interactionPointsIn3D[:,1]] = 1

        # Step 5 - Get distanceMap
        # interactionDistanceMapVolume = getGaussianDistanceMap(interactionPointsVolume, distZ=3, sigma=0.03, view=view, show=False, sliceCount=11) # for (64,64,64
        # print (f" - [getDistMapVolume()] sliceId={sliceId}, view={view}, interactionType={interactionType}, interactionClass={interactionClass}")
        if str(patientId) == 'CHMR040asfasdf': # and int(sliceId) == 65 and view == config.KEY_AXIAL:
            interactionDistanceMapVolume = getGaussianDistanceMap(interactionPointsVolume, distZ=3, sigma=0.005, patientId=patientId, sliceId=sliceId, view=view, show=True, sliceCount=11)
            pdb.set_trace()
        else:
            interactionDistanceMapVolume = getGaussianDistanceMap(interactionPointsVolume, distZ=3, sigma=0.005, patientId=patientId, sliceId=sliceId, view=view, show=False, sliceCount=None) # for (144,144,144)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return interactionDistanceMapVolume, interactionType, interactionClass

def getDistMapVolumeForViews(requestedPatient, requestedSliceIdx, volumeFailureFP, volumeFailureFN
                             , idsSortedAxial, idsSortedSagittal, idsSortedCoronal, sliceParams, verbose=False):
    """
    Params
    ------
    requestedPatient: str, patient Name
    requestedSliceIdx: int, slice id

    volumeFailureFP: [H,W,Depth], np.ndarray, containing 1s and 0s
    volumeFailureFN: [H,W,Depth], np.ndarray, containing 1s and 0s
    
    idsSortedAxial: [N], np.ndarray, containing sliceIds sorted by error area in descending order
    idsSortedSagittal: [N], np.ndarray, 
    idsSortedCoronal: [N], np.ndarray,
    sliceParams: dict, containing slice parameters
    """

    # Step 0.1 - Init
    timeForViews = time.perf_counter() # [time.time(), time.thread_time(), time.process_time()]
    meta = []
    interactionDistanceMapVolume = None
    interactionClass = None

    try:

        # Step 0.2 - Get vars
        interactionType = sliceParams[config.KEY_INTERACTION_TYPE]
        slicesPerView   = sliceParams[config.KEY_PERVIEW_SLICES]
        axialBool, sagittalBool, coronalBool = False, False, False # for debugging

        # Step 1 - Convert to numpy
        volumeFailureFP = np.array(volumeFailureFP, dtype=np.uint8)
        volumeFailureFN = np.array(volumeFailureFN, dtype=np.uint8)
        idsSortedAxial = np.array(idsSortedAxial, dtype=np.uint8)
        idsSortedSagittal = np.array(idsSortedSagittal, dtype=np.uint8)
        idsSortedCoronal = np.array(idsSortedCoronal, dtype=np.uint8)
        
        # Step 2.1 - For axial view
        if 0 <= requestedSliceIdx < 1*slicesPerView:
        # if 1*slicesPerView <= requestedSliceId < 2*slicesPerView: # [TODO: Just for debugging]
            axialBool = True
            axialIdx        = int(idsSortedAxial[:slicesPerView][requestedSliceIdx % 3])

            interactionDistanceMapVolume, interactionType, interactionClass  = getDistMapVolume(volumeFailureFP, volumeFailureFN, axialIdx, config.KEY_AXIAL, sliceParams, requestedPatient)
            meta = [requestedPatient, interactionType, interactionClass, config.KEY_AXIAL, axialIdx]

        # Step 2.2 - For sagittal view
        elif 1*slicesPerView <= requestedSliceIdx < 2*slicesPerView:
        # elif 0 <= requestedSliceId < 1*slicesPerView: # [TODO: Just for debugging]
        # elif 2*slicesPerView <= requestedSliceId < 3*slicesPerView: # [TODO: Just for debugging]
            sagittalBool = True
            sagittalIdx         = int(idsSortedSagittal[:slicesPerView][requestedSliceIdx % 3])

            interactionDistanceMapVolume, interactionType, interactionClass  = getDistMapVolume(volumeFailureFP, volumeFailureFN, sagittalIdx, config.KEY_SAGITTAL, sliceParams, requestedPatient)
            meta = [requestedPatient, interactionType, interactionClass, config.KEY_SAGITTAL, sagittalIdx]
        
        # Step 2.3 - For coronal view
        elif 2*slicesPerView <= requestedSliceIdx < 3*slicesPerView:
        # elif 0 <= requestedSliceId < 1*slicesPerView: # [TODO: Just for debugging]
            coronalBool = True
            coronalIdx         = int(idsSortedCoronal[:slicesPerView][requestedSliceIdx % 3])

            interactionDistanceMapVolume, interactionType, interactionClass  = getDistMapVolume(volumeFailureFP, volumeFailureFN, coronalIdx, config.KEY_CORONAL, sliceParams, requestedPatient)
            meta = [requestedPatient, interactionType, interactionClass, config.KEY_CORONAL, coronalIdx]


    except:
        traceback.print_exc()
        pdb.set_trace()
    
    if verbose: print (f" - [utils.getDistMapVolumeForViews()] timeForViews={time.perf_counter()-timeForViews:.2f}")
    return meta, interactionClass, interactionDistanceMapVolume

############################################
# PATIENT DATA
############################################

def getPatientNamesAndPaths(dirParams, verbose=False): 

        try:
            
            # Step 0 - Init
            tFiles = time.time()
            dirDataOG = dirParams[config.KEY_DIR_DATA_OG]
            fileExt   = dirParams[config.KEY_EXT]
            regexCT   = dirParams[config.KEY_REGEX_CT]
            regexStrFormatCT = str(dirParams[config.KEY_STRFMT_CT]).replace('.','\\.').replace('{}', '(.+)')

            # Step 1 - Get patient names and paths
            patientPaths = {}
            patientIdxToNameObj = {}
            patientIdx         = 0
            for fileName in Path(dirDataOG).rglob('*' + regexCT + '*'):
                if fileName.suffix == fileExt:
                    match = re.search(regexStrFormatCT, fileName.parts[-1])
                    if match:
                        patientName = match.group(1)    
                        patientIdxToNameObj[patientIdx] = patientName
                        patientIdx += 1
                        patientPaths[patientName] = {
                            config.KEY_CT  : Path(dirDataOG) / dirParams[config.KEY_STRFMT_CT].format(patientName),
                            config.KEY_PET : Path(dirDataOG) / dirParams[config.KEY_STRFMT_PET].format(patientName),
                            config.KEY_GT  : Path(dirDataOG) / dirParams[config.KEY_STRFMT_PRED].format(patientName),
                            config.KEY_PRED: Path(dirDataOG) / dirParams[config.KEY_STRFMT_GT].format(patientName)
                        }
            
            assert len(patientPaths) > 0, ' - [utils.getPatientNamesAndPaths()] No patients found in {}'.format(dirDataOG)
            assert len(patientIdxToNameObj) * 4 == len(list(Path(dirDataOG).glob('*' + fileExt))), 'Number of patients and number of files do not match'
            if verbose: print (' - [utils.getPatientNamesAndPaths()] Time taken: ', round(time.time() - tFiles,2))

        except:
            traceback.print_exc()
            pdb.set_trace()
        
        return patientPaths, patientIdxToNameObj

def generatePatientSlicesIndexes(patientIdxToNameObj, count, sliceParams):
        """
        To be called prior to every epoch for randomization purposes
         - Copies of this are sent to every worker

        Note
            - patientIds = not real patientIds, but the index of the patient in the list of patients in the directory
        """

        # Step 0 - Init
        patientIdxToSliceIdxsObjs = [dict({}) for _ in range(count)]
        patientIdxs = list(patientIdxToNameObj.keys())
        np.random.shuffle(patientIdxs)
        patientIdxsForWorkers = np.array_split(patientIdxs, count)

        # Step 1 - Generate slice indexes for each patient
        for workerId, patientIdxs in enumerate(patientIdxsForWorkers):
            for patientIdx in patientIdxs:
                sliceIdxs = np.arange(0, sliceParams[config.KEY_PERVIEW_SLICES] * 3).tolist()
                np.random.shuffle(sliceIdxs)
                patientIdxToSliceIdxsObjs[workerId][patientIdx] = list(sliceIdxs)
        
        return patientIdxToSliceIdxsObjs

def getPatientData(thisPatientPaths, sliceParams, patientName, verbose=False):
    """
    Params
    ------
    patientPaths: dict, containing paths to patient data
    sliceParams: dict, containing slice parameters
    """

    # Step 0 - Init
    tPatient = time.time()
    xCT, xPET, yGT, yPred, errorFP, errorFN, idsSortedAxial, idsSortedSagittal, idsSortedCoronal = None, None, None, None, None, None, None, None, None 

    try:
        
        print (f" - [utils.getPatientData()] patientName={patientName}")
        arrayCT, _ = nrrd.read(thisPatientPaths[config.KEY_CT])
        if len(arrayCT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] CT volume is not 3D')        

        # Step 1.2 - Get PET
        arrayPT, _ = nrrd.read(thisPatientPaths[config.KEY_PET])
        if len(arrayPT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] PET volume is not 3D')

        # Step 1.3 - Get GT
        arrayMaskGT, _  = nrrd.read(thisPatientPaths[config.KEY_GT])
        if len(arrayMaskGT.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] GT volume is not 3D')
        arrayMaskGTLabel = getMaskForLabel(arrayMaskGT, sliceParams[config.KEY_LABEL])

        # Step 1.4 - Get Pred
        arrayMaskPred, _ = nrrd.read(thisPatientPaths[config.KEY_PRED])
        if len(arrayMaskPred.shape) != 3: print (' - [PointAndScribbleDataset._getPatientData()] Pred volume is not 3D')
        arrayMaskPredLabel = getMaskForLabel(arrayMaskPred, sliceParams[config.KEY_LABEL])

        # Step 1.5 - Get failure areas
        failureAreasTorch, failureAreasFalseNegativesTorch, failureAreasFalsePositivesTorch = getFailureAreasTorch(arrayMaskGTLabel, arrayMaskPredLabel, sliceParams[config.KEY_KSIZE_SEGFAILURE])
        if failureAreasTorch is None: 
            return xCT, xPET, yGT, yPred, errorFP, errorFN, idsSortedAxial, idsSortedSagittal, idsSortedCoronal

        # Step 1.6 - Get failure areas (sorted in descending order of area)
        idsSortedAxial, idsSortedSagittal, idsSortedCoronal, failureAreaSumAxial, failureAreaSumSagittal, failureAreaSumCoronal = getFailureAreasStatsByView(failureAreasTorch)

        xCT     = arrayCT
        xPET    = arrayPT
        yGT     = np.array(arrayMaskGTLabel, dtype=np.uint8)
        yPred   = np.array(arrayMaskPredLabel, dtype=np.uint8)
        errorFP = np.array(failureAreasFalsePositivesTorch, dtype=np.uint8)
        errorFN = np.array(failureAreasFalseNegativesTorch, dtype=np.uint8) 

    except:
        traceback.print_exc()
        pdb.set_trace()

    if verbose: print (f" - [utils.getPatientData()][{patientName}] timeForPatient={time.time()-tPatient:.2f}")

    return xCT, xPET, yGT, yPred, errorFP, errorFN, idsSortedAxial, idsSortedSagittal, idsSortedCoronal

############################################
# FAILURE AREA GENERATION
############################################

def getMaskForLabel(arrayMask, label):

    arrayMaskOfLabel = None
    try:
        arrayMaskOfLabel = np.zeros(arrayMask.shape)
        arrayMaskOfLabel[arrayMask == label] = 1
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return arrayMaskOfLabel

def doErosionDilationTorch(volume, kSize=(3,3,1)):
    """
    volume: [H,W,Depth], np.ndarray
    kSize: (h,w,depth)
    """
    
    volTorchErodedDilated = None
    try:

        # Step 0 - Init
        assert type(volume) == np.ndarray, " - [doErosionDilationTorch()] volume must be np.ndarray"
        assert len(volume.shape) == 3, " - [doErosionDilationTorch()] volume must be [H,W,Depth], but it is {}".format(volume.shape)

        # Step 1 - Perform erosion-dilation
        volTorch        = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(volume), dim=-1), dim=0) # [1,H,W,Depth,1]
        volTorchEroded  = -torch.nn.functional.max_pool3d(-volTorch, kernel_size=kSize, stride=1, padding=1)
        volTorchErodedDilated = torch.nn.functional.max_pool3d(volTorchEroded, kernel_size=kSize, stride=1, padding=1)[0,:,:,:,0]
        volTorchErodedDilated = volTorchErodedDilated.byte()

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return volTorchErodedDilated # [H,W,Depth]
    
def getFailureAreasTorch(arrayMaskGTOfLabel, arrayMaskPredOfLabel, kSize=(3,3,1)):
    """
    Params
    ------
    arrayMaskGTOfLabel: [H,W,Depth] contains 0s and 1s, np.array or torch.tensor
    arrayMaskPredOfLabel: [H,W,Depth] contains 0s and 1s, np.array or torch.tensor
    """


    failureAreasTorch = None
    failureAreasFalseNegativesTorch, failureAreasFalsePositivesTorch = None, None

    try:

        # Step 0 - Init
        # assert np.unique(arrayMaskGTOfLabel).tolist() == [0,1], " - [getErrorAreas()] arrayMaskGTOfLabel must contain 0s and 1s only" # [NOTE: will slow down training, uncomment when publishing code]
        # assert arrayMaskGTOfLabel.shape == arrayMaskPredOfLabel.shape, " - [getErrorAreas()] arrayMaskGTOfLabel and arrayMaskPredOfLabel must have the same shape"
        # assert np.unique(arrayMaskPredOfLabel).tolist() == [0,1], " - [getErrorAreas()] arrayMaskPredOfLabel must contain 0s and 1s only" # [NOTE: will slow down training, uncomment when publishing code]

        # Step 1 - Get error areas
        errorAreas = np.zeros(arrayMaskGTOfLabel.shape)
        errorAreas[arrayMaskGTOfLabel != arrayMaskPredOfLabel] = 1

        # Step 1 - Perform erosion-dilation
        failureAreasTorch = doErosionDilationTorch(errorAreas, kSize=kSize) # [H,W,Depth]
        # if failureAreasTorch is not None: 
        #     failureAreas = failureAreasTorch.numpy()
        
        # Step 2 - Return false positives and false negative arrays
        arrayMaskGTOfLabelFgd = arrayMaskGTOfLabel
        arrayMaskGTOfLabelBgd = 1 - arrayMaskGTOfLabel

        failureAreasFalsePositivesTorch = failureAreasTorch * arrayMaskGTOfLabelBgd
        failureAreasFalseNegativesTorch = failureAreasTorch * arrayMaskGTOfLabelFgd

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return failureAreasTorch, failureAreasFalseNegativesTorch, failureAreasFalsePositivesTorch

def getFailureAreasStatsByView(failureAreas):
    """
    Params
    ------
    failureAreas: [H,W,Depth], torch.tensor
    """
    idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, failureAreaSumAxial, failureAreaSumSagittal, failureAreaSumCoronal = None, None, None, None, None, None

    try:
        
        # Step 1 - In axial, sagittal and coronal views, get the error areas 
        failureAreaSumAxial    = torch.sum(failureAreas, axis=(0,1))
        failureAreaSumSagittal = torch.sum(failureAreas, axis=(1,2))
        failureAreaSumCoronal  = torch.sum(failureAreas, axis=(0,2))        

        # Step 2.1 - Sort and get idxs
        idxsSortedAxial    = torch.flip(torch.argsort(failureAreaSumAxial.flatten()), dims=[0]) # [::-1 does not work in torch]
        idxsSortedSagittal = torch.flip(torch.argsort(failureAreaSumSagittal.flatten()), dims=[0])
        idxsSortedCoronal  = torch.flip(torch.argsort(failureAreaSumCoronal.flatten()), dims=[0])

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return idxsSortedAxial, idxsSortedSagittal, idxsSortedCoronal, failureAreaSumAxial, failureAreaSumSagittal, failureAreaSumCoronal

############################################
# PLOTTING
############################################

def convertColorMapWithLinearTransparency(cmap):

    cmapNew = None 

    try:
        import matplotlib
        assert type(cmap) == matplotlib.colors.LinearSegmentedColormap 
        
        cmapNew = cmap(np.arange(cmap.N))
        cmapNew[:,-1] = np.linspace(0, 1, cmap.N)

        import matplotlib.colors as mcolors
        cmapNew = mcolors.ListedColormap(cmapNew)

    except:
        traceback.print_exc()
        pdb.set_trace()
        cmapNew = plt.cm.Oranges  # [plt.cm.jet]

    return cmapNew

def convertColorMapWithPowerLawTransparency(cmap, power=0.5):

    cmapNew = None 

    try:
        import matplotlib
        assert type(cmap) == matplotlib.colors.LinearSegmentedColormap 
        
        cmapNew = cmap(np.arange(cmap.N))
        cmapNew[:,-1] = np.linspace(0, 1, cmap.N)**power

        import matplotlib.colors as mcolors
        cmapNew = mcolors.ListedColormap(cmapNew)

    except:
        traceback.print_exc()
        pdb.set_trace()
        cmapNew = plt.cm.Oranges  # [plt.cm.jet]

    return cmapNew

def getYellowColorMap(show=False):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Define the colormap colors (you can adjust these as desired)
    # colors = ["gold", "yellow",  "lightyellow"]
    # colors = ["gold", "yellow",  "white"]
    colors = ["white", "gold"]

    # Create a LinearSegmentedColormap from the list of colors
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

    if show:
        # Generate some example data (you can replace this with your own data)
        x_values = np.linspace(0, 1, 100)
        y_values = np.sin(2 * np.pi * x_values)  # Example function

        # Plot the data using the colormap
        plt.figure(figsize=(8, 6))
        plt.scatter(x_values, y_values, c=x_values, cmap=cmap, s=50)
        plt.colorbar(label="x values")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Sequential Yellow Colormap")
        plt.grid(True)
        plt.show()
    
    return cmap

def convertColopMapWithZeroAsTransparent(cmap):
    """
    Params
    ------
    cmap: matplotlib.colors.LinearSegmentedColormap
    """

    cmapNew = None

    try:
        
        import matplotlib
        assert type(cmap) == matplotlib.colors.LinearSegmentedColormap 
        
        cmapNew = cmap(np.arange(cmap.N))
        cmapNew[0,3] = 0

        import matplotlib.colors as mcolors
        cmapNew = mcolors.ListedColormap(cmapNew)


    except:
        traceback.print_exc()
        pdb.set_trace()
        cmapNew = plt.cm.Oranges  # [plt.cm.jet]
    
    return cmapNew

############################################
# VISUALIZATION
############################################

def showFunc(x1, x2, y1, y2, z1, z2, meta, sliceCount=7, contourWidth=0.25):
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
            
            f,axarr = plt.subplots(batchSize, 2+sliceCount, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(10,4))
            if batchSize == 1: axarr =  np.expand_dims(axarr, 0)
            f.subplots_adjust(hspace=0.1, wspace=0.1)
            cmapPet = convertColorMapWithLinearTransparency(plt.cm.Oranges)
            cmapDistMapFgd = convertColopMapWithZeroAsTransparent(plt.cm.Blues)
            cmapDistMapBgd = convertColopMapWithZeroAsTransparent(getYellowColorMap())

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
                sliceId          = int(meta[4][batchId]) # in that view)
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
            
            plt.suptitle('Foreground (Blue) and Background (Yellow) Distance Maps')
            plt.show(block=False)
            # pdb.set_trace()
        
        except:
            traceback.print_exc()
            pdb.set_trace()


############################################
# MAIN
############################################

if __name__ == "__main__":

    
    # Test 1 - getDistanceMap()
    if 1:
        # binaryMask = torch.zeros(144, 144, 144); sliceId=30; view=config.KEY_AXIAL
        binaryMask = np.zeros((144, 144, 144)); sliceId=30; view=config.KEY_SAGITTAL
        # binaryMask = torch.zeros(64, 64, 64); sliceId=30
        binaryMaskWithScribble = getRandomBezierCurveOnMask(binaryMask, sliceId, view, numPoints=3)

        # using scipy
        if 1:
            
            # distZ, sigma = 3, 0.03 # best parameter choice (for (64,64,64))
            # distZ, sigma = 3, 0.02 # nope for (144,144,144)
            # distZ, sigma = 1, 0.02 # nope for (144,144,144)
            # distZ, sigma = 1, 0.05 # nope for (144,144,144)
            # distZ, sigma = 3, 0.05 # nope for (144,144,144)
            # distZ, sigma = 3, 0.01 # meh for (144,144,144) --> 5 slices
            # distZ, sigma = 1, 0.01 # nope for (144,144,144) 
            distZ, sigma = 3, 0.005 # WORKS for (144,144,144) !!!
            gaussianDistanceMap = getGaussianDistanceMap(binaryMaskWithScribble, distZ, sigma, view=view, patientId='Test', sliceId=sliceId, show=True, sliceCount=11)

            # from timeit import timeit
            # totalTime = timeit(lambda: getGaussianDistanceMap(binaryMaskWithScribble, distZ, sigma, view, show=False, sliceCount=11), number=500)
            # print (f"Average time is {totalTime / 100:.2f} seconds")    
        
        elif 1:
            pass

            
        pdb.set_trace()

"""
https://docs.monai.io/en/latest/transforms.html#distancetransformedt
"""

"""
Data Sources
- trial1: Z:\2021_HECKTOR_HNTumorAuto\_models\FocusNetV4ResV2-LR10e3I15__ValCenterFold1-B2-NewWindowing-Contrast3-Flip035__Th09CEF095B005__seed42
"""