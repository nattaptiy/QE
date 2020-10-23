from os import makedirs, path
import torch
from transformers import AutoTokenizer
from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from config.mlqe import MODEL_PATH, MODEL_TYPE, GRAD_CLIP_NORM, BEST_MODEL_FOLDER_PREFIX, RESOURCE_PATH,RESOURCE_TRAIN_PATH, \
    RESOURCE_DEV_PATH, RESOURCE_TEST_PATH
from data.mlqe.data_manager import DataManager
from models.mlqe.model import LightningModel
from utils.calc_pearsonr import calc_pearsonr
from utils.command import CommandMeta

seed_everything(42)


class Trainer(CommandMeta):
    def __init__(self,
                 model_type=MODEL_TYPE,
                 gpus="6",
                 model_base_path=MODEL_PATH,
                 best_model_folder_prefix=BEST_MODEL_FOLDER_PREFIX,
                 grad_clip_norm=GRAD_CLIP_NORM,
                 resource_train_path=RESOURCE_TRAIN_PATH,
                 resource_dev_path=RESOURCE_DEV_PATH,
                 resource_test_path=RESOURCE_TEST_PATH):
        super().__init__()
        self.gpus = gpus
        self.model_type = model_type
        self.grad_clip_norm = grad_clip_norm
        self.model_base_path = model_base_path
        self.best_model_folder_prefix = best_model_folder_prefix
        self.resource_train_path = resource_train_path
        self.resource_dev_path = resource_dev_path
        self.resource_test_path = resource_test_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.data_manager = DataManager(tokenizer=self.tokenizer)
        self.train_dataloader = self.data_manager.load_file(self.resource_train_path, manage_cache=False)
        self.dev_dataloader = self.data_manager.load_file(self.resource_dev_path, manage_cache=False)
        self.test_dataloader = self.data_manager.load_file(self.resource_test_path, manage_cache=False, do_eval=True)
        #self.test_dataloader = self.data_manager.load_file(self.resource_dev_path, manage_cache=False, do_eval=True)

    def train_single(self, train_dataloader, dev_dataloader):
        finetuner = LightningModel(train_dataloader=train_dataloader,
                                   dev_dataloader=dev_dataloader,
                                   model_type=self.model_type,
                                   model_path=self.model_base_path,
                                   test_dataloader=self.test_dataloader)
        early_stopping_callback = EarlyStopping(monitor='val_pearsonr', patience=10, mode="max")
        lr_logger_callback = LearningRateLogger()

        makedirs(self.model_base_path, exist_ok=True)
        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.model_base_path,
            monitor="val_pearsonr", save_top_k=1)
        trainer = pl.Trainer(gpus=self.gpus,
                             max_epochs=100,
                             default_root_dir=self.model_base_path,
                             gradient_clip_val=self.grad_clip_norm,
                             log_save_interval=100,
                             val_check_interval=0.25,
                             checkpoint_callback=model_checkpoint_callback,
                             early_stop_callback=early_stopping_callback,
                             callbacks=[lr_logger_callback],
                             progress_bar_refresh_rate=1)
        trainer.fit(finetuner)
        trainer.test()

    def load_lightning_model_from_path(self,
                                       pretrained_path,
                                       train_dataloader=None,
                                       dev_dataloader=None,
                                       test_dataloader=None):
        model = LightningModel(
            model_type=self.model_type,
            pretrained_model_path=pretrained_path,
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            test_dataloader=test_dataloader)
        return model

    def train(self):
        self.train_single(train_dataloader=self.train_dataloader, dev_dataloader=self.dev_dataloader)

    def predict_dataloader(self, model, dataloader):
        preds,  labels = list(), list()
        with torch.no_grad():
            model.eval()
            for batch in dataloader:
                batch = [item.to("cuda:6") for item in batch]
                model.to("cuda:6")
                pred = model.predict(batch)
                pred = pred.to("cpu")
                preds.append(pred)
                inputs = self.data_manager.fix_batch(batch)
                label = inputs["labels"].to("cpu")
                labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        return preds,  labels

    def test(self):
        model = self.load_lightning_model_from_path(pretrained_path=path.join(self.model_base_path, self.best_model_folder_prefix),
                                                    train_dataloader=self.train_dataloader,
                                                    dev_dataloader=self.dev_dataloader,
                                                    test_dataloader=self.test_dataloader)
        preds, labels = self.predict_dataloader(model, self.test_dataloader)
        print("TEST RESULT")
        print(calc_pearsonr(preds, labels))
        print("TEST END")
        return preds, labels

    def exec(self):
        self.train()
        self.test()


if __name__ == '__main__':
    for MODEL_TYPE in ["xlm-roberta-large"]:
        print(MODEL_TYPE)
        finetuner = Trainer(model_type=MODEL_TYPE)
        finetuner.exec()

