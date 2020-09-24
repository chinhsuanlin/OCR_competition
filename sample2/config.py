import argparse
import string
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=144)
parser.add_argument('--height', type=int, default=35)
parser.add_argument('--nb_channels', type=int, default=1)
parser.add_argument('--label_len', type=int, default=7)
parser.add_argument('--batch_size', type=int, default = 150)
parser.add_argument('--model', type=str, default='CRNN_STN', choices=['CRNN_STN', 'CRNN'])
parser.add_argument('--conv_filter_size', type=int, nargs=3, default=[64, 128, 256, 256, 512, 512, 512])
#[64, 128, 256, 256, 512, 512, 512]
parser.add_argument('--lstm_nb_units', type=int, nargs=2, default=[128, 128])
parser.add_argument('--timesteps', type=int, default=25)
parser.add_argument('--dropout_rate', type=float, default=0.25)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_reduction_factor', type=float, default=0.1)
parser.add_argument('--nb_epochs', type=int, default = 500)
parser.add_argument('--val_iter_period', type=int, default = 100000)
parser.add_argument('--gpus', type=int, nargs='*', default=[0])
parser.add_argument('--nb_workers', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--resume_training', type=bool, default=True)

parser.add_argument('--characters', type=str, default = '0123456789' + string.ascii_uppercase + '-' ) 
#parser.add_argument('--characters', type=str, default = '0123456789' + string.ascii_lowercase  + '-') 
parser.add_argument('--base_dir', type=str, default='./data/')
#D:\\DataSet\\Visual Geometry Group\\mnt\\ramdisk\\max\\90kDICT32px\\
#D:\\DataSet\\Visual Geometry Group\\demo\\
#D:\\DataSet\\OCR\\
parser.add_argument('--output_dir', type=str, default='result')
parser.add_argument('--save_best_only', type=bool, default=True)
parser.add_argument('--training_model_cp_filename', type=str, default='model.{epoch:03d}-{loss:.2f}.hdf5')
parser.add_argument('--prediction_model_cp_filename', type=str, default='prediction_model.{epoch:03d}.hdf5')
parser.add_argument('--load_model_path', type=str, default='prediction_model.hdf5')
parser.add_argument('--tb_log', type=str, default='tb_log')

cfg = parser.parse_args()
