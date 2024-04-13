# Import private modules
import src.config as config

# Import public modules
from pathlib import Path

DIR_THIS = Path(__file__).resolve().parent.absolute() # <projectRoot>/src/trainerParams
DIR_ROOT = DIR_THIS.parent.parent.absolute() # <projectRoot>
DIR_DATA = DIR_ROOT / '_data'

params = {
        config.KEY_EXP_NAME: 'exp1',
        config.KEY_DATALOADER_PARAMS_GLOBAL:{
            config.KEY_TRAIN: {
                config.KEY_DIR_PARAMS: {
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
                },
                config.KEY_SLICE_PARAMS: {
                    config.KEY_PERVIEW_SLICES: 5,
                    config.KEY_KSIZE_SEGFAILURE: (3,3,3),
                    config.KEY_LABEL: 1,
                    config.KEY_INTERACTION_TYPE: [config.KEY_INTERACTION_SCRIBBLES], # [config.KEY_INTERACTION_POINTS, config.KEY_INTERACTION_SCRIBBLES]
                    config.KEY_SCRIBBLE_TYPE: [config.KEY_SCRIBBLE_MEDIAL_AXIS] # config.KEY_SCRIBBLE_RANDOM, config.KEY_SCRIBBLE_MEDIAL_AXIS
                },
                config.KEY_DATALOADER_PARAMS : {
                    config.KEY_PATIENTLOADER: {
                        config.KEY_PATIENTLOADER_WORKERS: 1 # TODO: need to write logic for this
                        , config.KEY_PATIENTLOADER_BUFFER: 4
                        , config.KEY_PATIENTLOADER_DIMS: (144,144,144)
                    }
                    , config.KEY_ANNOTATIONLOADER: {    
                        config.KEY_ANNOTATIONLOADER_WORKERS: 4 
                        , config.KEY_EPOCHS: 2
                        , config.KEY_BATCH_SIZE:4
                        , config.KEY_QUEUE_MAXLEN: 30
                        , config.KEY_DEVICE: config.deviceGPU
                    }
                }
            },
            config.KEY_VAL: {

            }
        },
        config.KEY_MODEL_PARAMS: {
            config.KEY_NEURALNET: config.KEY_UNET_V1,
            config.KEY_LR: 0.001,
            config.KEY_LOSSES: [config.KEY_LOSS_DICE],    # config.KEY_LOSS_PERCEPTUAL, config.KEY_LOSS_SSIM, config.KEY_LOSS_JUKEBOX
            config.KEY_MODEL_SAVE_RATE: 10
        }
    }