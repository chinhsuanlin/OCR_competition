#%%
import os
import argparse
import string
from tqdm import tqdm
import numpy as np
import cv2
import keras.backend as K
from keras.models import model_from_json, load_model
import copy
from utils import pad_image, resize_image, create_result_subdir
from STN.spatial_transformer import SpatialTransformer
from models import CRNN, CRNN_STN

import segmentation
#import rotation
#import tranditaional
import os
import pandas as pd
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=r'D:\Lycoris\final\sample6\Model\result\002\prediction_model.010.hdf5')
parser.add_argument('--data_path', type=str, default='./test/')
parser.add_argument('--gpus', type=int, nargs='*', default=[0])
parser.add_argument('--characters', type=str, default = '0123456789' + string.ascii_uppercase + '-'  )
parser.add_argument('--label_len', type=int, default=7)
parser.add_argument('--nb_channels', type=int, default=1)
parser.add_argument('--width', type=int, default=135)
parser.add_argument('--height', type=int, default=90)
parser.add_argument('--model', type=str, default='CRNN_STN', choices=['CRNN_STN', 'CRNN'])
parser.add_argument('--conv_filter_size', type=int, nargs=7, default=[64, 128, 256, 256, 512, 512, 512])
parser.add_argument('--lstm_nb_units', type=int, nargs=2, default=[128, 128])
parser.add_argument('--timesteps', type=int, default = 24)
parser.add_argument('--dropout_rate', type=float, default=0.25)
cfg = parser.parse_args()
#%%
def set_gpus():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)[1:-1]

def create_output_directory():
    os.makedirs('eval', exist_ok=True)
    output_subdir = create_result_subdir('eval')
    print('Output directory: ' + output_subdir)
    return output_subdir

def collect_data():
    if os.path.isfile(cfg.data_path):
        return [cfg.data_path]
    else:
        files = [os.path.join(cfg.data_path, f) for f in os.listdir(cfg.data_path) if f[-4:] in ['.jpg', '.JPG', '.png', '.PNG', 'jpeg']]
        return files

def load_image(img_path):
    if cfg.nb_channels == 1:
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(img_path)    

def preprocess_image(img):
    if img.shape[1] / img.shape[0] < 6.4:
        img = pad_image(img, (cfg.width, cfg.height), cfg.nb_channels)
    else:
        img = resize_image(img, (cfg.width, cfg.height))
    if cfg.nb_channels == 1:
        img = img.transpose([1, 0])
    else:
        img = img.transpose([1, 0, 2])
    img = np.flip(img, 1)
    img = img / 255.0
    if cfg.nb_channels == 1:
        img = img[:, :, np.newaxis]
    return img

def predict_text(model, img):
    y_pred = model.predict(img[np.newaxis, :, :, :])
    shape = y_pred[:, 2:, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
    ctc_out = K.get_value(ctc_decode)[:, :cfg.label_len]
    result_str = ''.join([cfg.characters[c] for c in ctc_out[0]])
    result_str = result_str.replace('-', '')
    return result_str

def evaluate(model, data, output_subdir):
    if len(data) == 1:
        evaluate_one(model, data)
    else:
        evaluate_batch(model, data, output_subdir)

def evaluate_one(model, data):
    img = load_image(data[0])
    img = preprocess_image(img)
    result = predict_text(model, img)
    print('Detected result: {}'.format(result))

def evaluate_batch(model, data, output_subdir):
    for filepath in tqdm(data):        
        img = load_image(filepath)
        img = preprocess_image(img)
        result = predict_text(model, img)
        output_file = os.path.basename(filepath)
        output_file = output_file[:-4] + '.txt'
        with open(os.path.join(output_subdir, output_file), 'w') as f:
            f.write(result)
def collet_txts(data):
    txts = copy.deepcopy(data)
    for index, val in enumerate(data):
        txts[index] =  output_subdir + '/' +  val.replace('./test/','').replace('png', '') + 'txt'
    return (txts)

def gencsv(output_subdir):
    cells = []
    labels = []
    files = os.listdir(output_subdir)
    for val in files:
        cells.append( val.replace('.txt', '') )
        with open (output_subdir + '\\' + val, 'r') as f:
            string = f.readline()
            string = string.replace("a", "/").replace("b", "\\")
        labels.append(string)
    table = pd.DataFrame( {'Number' : cells, 'Content' : labels})
    table.to_excel('./Team046_06.xlsx')
#%%
if __name__ == '__main__':
    #set_gpus()
#    segg()
    output_subdir = create_output_directory()
    data = collect_data()
    txts = collet_txts(data)
    _, model = CRNN_STN(cfg)
    model.load_weights(cfg.model_path)
    evaluate(model, data, output_subdir)
#    tranditaional.post_process(data, txts)
    gencsv(output_subdir)
