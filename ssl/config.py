from easydict import EasyDict as edict
import yaml

def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

config = edict()

config.GPUS = '0'
config.PRINT_FREQ = 100

config.OUTPUT_DIR = './results'
config.LOG_DIR = './results'

config.FIX_VIEW_ID = -1
config.IMG_SIZE = (224, 224)
config.CROP_BOX = True
config.ZOOM_OUT_RATIO = 1

# dataset
config.DATASET = edict()
config.DATASET.TRAIN_DATASET = 'wildtrack'
config.DATASET.TEST_DATASET = 'wildtrack'
config.DATASET.VIEW_IDS = [0,1,2,3,4,5,6]
config.DATASET.SAMPLING_RANGE = [5, 20]
config.DATASET.PROMPT_MODE = 'box'
config.DATASET.ALPHA = 0.1

# training
config.TRAIN_GPUS = '0'

config.TRAIN = edict()
config.TRAIN.BATCH_SIZE = 1
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 200
config.TRAIN.RESUME = False
config.TRAIN.CKP_PATH = ''
config.TRAIN.LR = 1e-4
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [150]
config.TRAIN.PSEUDO_NEG = False
config.TRAIN.TRAIN_MODE = 'triplet'
config.TRAIN.EDGE_ASSOCIATION = False
config.TRAIN.REID = False
config.TRAIN.GT_BOX = True
config.TRAIN.THRESH = 0.0
config.TRAIN.FULLY_SUPERVISED = False

config.TEST = edict()
config.TEST.THRESH = 0.0
config.TEST.CKP_PATH = ''
config.TEST.REID = False
config.TEST.VIS = False