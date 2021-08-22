# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import argparse

from data import DInterface
from model import  MInterface
import pytorch_lightning as pl


def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--dataset_path',
                        default=r'/workspace/Dataset')  # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--category', default='carpet')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)  # 256
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.001)
    parser.add_argument('--project_root_path',
                        default=r'/workspace/result')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args, gpus=1, max_epochs=args.num_epochs, default_root_dir=os.path.join(args.project_root_path, args.category))
    if args.phase == 'train':
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=data_module)
    elif args.phase == 'test':
        trainer.test(model=model, datamodule=data_module)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
