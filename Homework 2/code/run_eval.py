import os
import os.path as osp
from argparse import ArgumentParser
import yaml

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from modules import MlpClassifier
from modules.feature_data_eval import FeatureDataModule_eval


import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image


def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('name')
    parser.add_argument('version_name')
    parser = FeatureDataModule_eval.add_argparse_args(parser)
    parser = MlpClassifier.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--earlystop_patience', type=int, default=15)
    parser = ArgumentParser(parents=[parser])
    parser.set_defaults(accelerator='gpu', devices=1,
                        default_root_dir=osp.abspath(
                            osp.join(osp.dirname(__file__), '../data/mlp')))
    args = parser.parse_args(argv)


    return args


def main(args):


    with open(osp.join(args.default_root_dir, args.name, args.version_name, 'hparams.yaml')) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    args.num_features = hparams['num_features']
    args.feature_dir = hparams['feature_dir']
    args.learning_rate = hparams['learning_rate']
    args.train_val_list_file = hparams['train_val_list_file']
    args.batch_size = hparams['batch_size']

    print(args.feature_dir, hparams['feature_dir'])
    


    data_module = FeatureDataModule_eval(args)
    model = MlpClassifier(args)
    # logger = TensorBoardLogger(args.default_root_dir, args.name)
    chk_file_list = os.listdir(osp.join(args.default_root_dir, args.name, args.version_name, 'checkpoints'))
    chk_path = osp.join(args.default_root_dir, args.name, args.version_name, 'checkpoints', chk_file_list[-1])

    model = MlpClassifier.load_from_checkpoint(chk_path)
    trainer = pl.Trainer.from_argparse_args(
        args)
    print('trained model loading successfully')

    val_df = pd.read_csv('data/labels/val.csv')

    predictions = trainer.predict(model=model, datamodule=data_module)
    df = data_module.test_df.copy()
    df.rename({'Category':'true'}, inplace=True, axis=1)
    df['pred'] = torch.concat(predictions).numpy()
    print(df.shape)
    cm = confusion_matrix(df['true'], df['pred'])
    with open(osp.join(args.default_root_dir, args.name, args.version_name, 'confusion_matrix.pkl'), 'wb') as f:
        pickle.dump(cm, f)
    print(cm)
    print(accuracy_score(df['true'], df['pred']))
    hm = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    hm.get_figure().savefig(osp.join(args.default_root_dir, args.name, args.version_name, 'confusion_matrix.jpg'))    
    # prediction_path = osp.join(logger.log_dir, 'test_prediction.csv')
    # df.to_csv(prediction_path, index=False)
    # print('Output file:', prediction_path)


if __name__ == '__main__':
    main(parse_args())
