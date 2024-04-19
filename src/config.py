import os
import torch

nameCPU = 'cpu'
nameGPU = 'cuda:0'
deviceCPU = torch.device("cpu")
deviceGPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PID   = os.getpid()

QUEUE_TIMEOUT = 5.0

KEY_DEVICE = 'device'
EXT_NRRD = '.nrrd'

KEY_EXT = 'ext'
KEY_DIR_DATA_OG = 'dir_og_data'

KEY_REGEX_CT = 'regex_ct'
KEY_REGEX_PET = 'regex_pet'
KEY_REGEX_GT = 'regex_gt'
KEY_REGEX_PRED = 'regex_pred'

KEY_STRFMT_CT = 'strfmt_ct'
KEY_STRFMT_PET = 'strfmt_pet'
KEY_STRFMT_GT = 'strfmt_gt'
KEY_STRFMT_PRED = 'strfmt_pred'

KEY_CT   = 'ct'
KEY_PET  = 'pet'
KEY_GT   = 'gt'
KEY_PRED = 'pred'
KEY_FAILURE_AREAS = 'failure_areas'
KEY_FAILURE_AREAS_FP = 'failure_areas_fp'
KEY_FAILURE_AREAS_FN = 'failure_areas_fn'

KEY_SORTED_AXIAL = 'sorted_axial'
KEY_SORTED_SAGITTAL = 'sorted_sagittal'
KEY_SORTED_CORONAL = 'sorted_coronal'

# Interaction {Type, Class}
KEY_INTERACTION_TYPE = 'interaction_type'
KEY_INTERACTION_POINTS = 'points'
KEY_INTERACTION_SCRIBBLES = 'scribbles'
KEY_INTERACTION_CLASS = 'interaction_class'
KEY_INTERACTION_FGD  = 'fgd'
KEY_INTERACTION_BGD  = 'bgd'

KEY_PERVIEW_SLICES = 'perview_slices'
KEY_KSIZE_SEGFAILURE = 'ksize_segfailure'
KEY_IGNORE_LABELS = 'ignore_label'
KEY_LABEL = 'label'

KEY_DIST_EUCLIDEAN = 'euclidean'
KEY_DIST_GEODESIC = 'geodesic'
KEY_DIST_GAUSSIAN = 'gaussian'
KEY_GAUSSIAN_SIGMA = 'gaussian_sigma'

KEY_AXIAL = 'axial'
KEY_SAGITTAL = 'sagittal'
KEY_CORONAL = 'coronal'

KEY_SCRIBBLE_TYPE = 'scribble_type'
KEY_SCRIBBLE_RANDOM = 'scribble_random'
KEY_SCRIBBLE_MEDIAL_AXIS = 'scribble_medial_axis'

COLOR_RED = 'red'
COLOR_GREEN = 'green'

KEY_PATIENTLOADER = 'patientloader'
KEY_PATIENTLOADER_WORKERS = 'patientloader_workers'
KEY_PATIENTLOADER_BUFFER = 'patientloader_buffer'
KEY_PATIENTLOADER_DIMS = 'patientloader_dims'

KEY_ANNOTATIONLOADER = 'annotationloader'
KEY_ANNOTATIONLOADER_WORKERS = 'annotation_loader_workers'

KEY_EPOCHS = 'epochs'
KEY_BATCH_SIZE = 'batch_size'

KEY_DATALOADER_PARAMS_GLOBAL = 'dataloader_global'
KEY_DATALOADER_PARAMS = 'dataloader'
KEY_DIR_PARAMS = 'dir_params'
KEY_SLICE_PARAMS = 'slice_params'
KEY_TRAIN = 'train'
KEY_VAL = 'valid'
KEY_TEST = 'test'

KEY_EXP_NAME = 'exp_name'

KEY_MODEL_PARAMS = 'model_params'

KEY_QUEUE_MAXLEN = 'queue_maxlen'

KEY_MODEL_PARAMS = 'model_params'
KEY_LR = 'lr'
KEY_UNET_V1 = 'unet_v1'

KEY_LOSSES = 'losses'
KEY_LOSS_BCE = 'bce'
KEY_LOSS_DICE = 'dice'

KEY_MODEL_SAVE_RATE = 'model_save_rate'
KEY_NEURALNET = 'neuralnet'

KEY_WORKERS = 'workers'
KEY_TIMELIST = 'timeList'
KEY_ITERPERSEC = 'iterPerSec'