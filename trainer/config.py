COLAB = True
GAN = True
LOAD = True
GZSL = True


DATASET_CUB = 'cub'
DATASET_AWA = 'awa'
DATASET_SUN = 'sun'
DATASET = DATASET_AWA


PARAMS = {
    DATASET_CUB: {
        'file_separator': ' ',
        'input_shape': 2048,
        'n_classes': 200,
        'n_unseen_classes': 50,
        'n_attributes': 312,
        'n_epochs': 70,
        'n_epochs_cls': 25,
        'batch_size': 64,
        'batch_size_cls': 64,
        'learning_rate': 0.0001,
        'learning_rate_cls': 0.001,
        'beta': 0.5,
        'syn_num': 300,
        'gradient_penalty_weight': 10,
        'cls_loss_weight': 0.01,
        'n_critic': 5,
        'latent_dim': 312
    },
    DATASET_AWA: {
        'file_separator': '\t',
        'input_shape': 2048,
        'n_classes': 50,
        'n_unseen_classes': 10,
        'n_attributes': 85,
        'n_epochs': 30,
        'n_epochs_cls': 25,
        'batch_size': 64,
        'batch_size_cls': 64,
        'learning_rate': 0.00001,
        'learning_rate_cls': 0.001,
        'beta': 0.5,
        'syn_num': 1800,
        'gradient_penalty_weight': 10,
        'cls_loss_weight': 0.01,
        'n_critic': 5,
        'latent_dim': 85
    },
    DATASET_SUN: {
        'file_separator': '\t',
        'input_shape': 2048,
        'n_classes': 717,
        'n_unseen_classes': 72,
        'n_attributes': 102,
        'n_epochs': 40,
        'n_epochs_cls': 25,
        'batch_size': 64,
        'batch_size_cls': 64,
        'learning_rate': 0.0002,
        'learning_rate_cls': 0.001,
        'beta': 0.5,
        'syn_num': 400,
        'gradient_penalty_weight': 10,
        'cls_loss_weight': 0.01,
        'n_critic': 5,
        'latent_dim': 102
    },
}

INPUT_SHAPE = PARAMS[DATASET]['input_shape']
N_CLASSES = PARAMS[DATASET]['n_classes']
N_UNSEEN_CLASSES = PARAMS[DATASET]['n_unseen_classes']
N_ATTRIBUTES = PARAMS[DATASET]['n_attributes']

LATENT_DIM = PARAMS[DATASET]['latent_dim']
N_CRITIC = PARAMS[DATASET]['n_critic']
N_EPOCHS_GAN = PARAMS[DATASET]['n_epochs']
N_EPOCHS_CLS = PARAMS[DATASET]['n_epochs_cls']
N_BATCH_GAN = PARAMS[DATASET]['batch_size']
N_BATCH_CLS = PARAMS[DATASET]['batch_size_cls']
LEARNING_RATE = PARAMS[DATASET]['learning_rate']
LEARNING_RATE_CLS = PARAMS[DATASET]['learning_rate_cls']
BETA = PARAMS[DATASET]['beta']

GRADIENT_PENALTY_WEIGHT = PARAMS[DATASET]['gradient_penalty_weight']
CLS_LOSS_WEIGHT = PARAMS[DATASET]['cls_loss_weight']
N_SAMPLES_GAN = N_UNSEEN_CLASSES * PARAMS[DATASET]['syn_num']

PATH = '/content/data' if COLAB else '../data'
FEATURES_PATH = PATH + '/' + DATASET + '/res101.mat'
ATTRIBUTES_PATH = PATH + '/' + DATASET + '/att_splits.mat'
FILE_SEPARATOR = PARAMS[DATASET]['file_separator']
MODEL_SAVE_PATH = 'cgan_generator.h5'
SAVE_MODEL_EVERY = 10
