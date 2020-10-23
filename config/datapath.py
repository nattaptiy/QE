from os import path
from utils.search_project_top import search_project_top

PROJECT_TOP = search_project_top(__file__)  # "/Users/reiven/Documents/Python/RewardExperiment"
DATAPATH = path.join(PROJECT_TOP, "data")
MODELPATH = path.join(PROJECT_TOP, "models")

BLEU_SH_PATH = path.join(PROJECT_TOP, "external_tools", "bleu.sh")