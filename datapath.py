from config.datapath import *
from os import path


MODEL_PATH = path.join(PROJECT_TOP, "models", "bert_mlqe")
TENSORBOARD_PATH = path.join(DATAPATH, "tensorboard", "BERT_mlqe")

RESOURCE_PATH = path.join(DATAPATH)
RAW_RESOURCE_PATH = path.join(DATAPATH, "raw", "mlqe")

lang='en-de'
RESOURCE_TRAIN_PATH = path.join(RESOURCE_PATH,lang+".train.tsv")
RESOURCE_DEV_PATH = path.join(RESOURCE_PATH,lang+".dev.tsv")
RESOURCE_TEST_PATH = path.join(RESOURCE_PATH,lang+".test.tsv")
# RAW_RESOURCE_TEST_PATH = path.join(RAW_RESOURCE_PATH, "test.shared-task.tsv")
