from pathlib import Path
from parse_argue import parser
from pytorch_lightning import Trainer
from train import Image_self_supervise
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    arg = parser.parse_args()

    if arg.train:
        print('----------train model----------')

        # ckpt_path = Path('../train_ckpt/resnet18_epoch=' + arg.epoch_idx + '.ckpt')
        # ckpt_path = './train_params/resnet50_epoch=06.ckpt'
        # model = Image_self_supervise.load_from_checkpoint(checkpoint_path = ckpt_path, map_location = None)
        model = Image_self_supervise()

        ckpt_dir_path = Path('../train_ckpt/')
        params_callback = ModelCheckpoint(dirpath = ckpt_dir_path, filename = 'resnet18'+'_{epoch:02d}', save_top_k = 3, mode = "min", monitor = "avg_loss")
            
        trainer = Trainer(callbacks = [params_callback], accelerator = "gpu", max_epochs = 250)
        # trainer = Trainer(accelerator = "gpu", min_epochs = 200, max_epochs = 250)
        trainer.fit(model)

    elif arg.valid:
        print('----------validate model----------')
        
        ckpt_path = Path('../train_ckpt/resnet18_epoch=' + arg.epoch_idx + '.ckpt')
        model = Image_self_supervise.load_from_checkpoint(checkpoint_path = ckpt_path, map_location = None)
        
        trainer = Trainer(accelerator = "gpu")
        trainer.validate(model)
        
    elif arg.test:
        print('----------test model----------')

        ckpt_path = Path('../train_ckpt/resnet18_epoch=' + arg.epoch_idx + '.ckpt')
        model = Image_self_supervise.load_from_checkpoint(checkpoint_path = ckpt_path, map_location = None)
        
        trainer = Trainer(accelerator = "gpu")
        trainer.test(model)