import numpy as np
import os
from PIL import Image 



train_dir = 'C:/Users/kordian.czyzewski/OneDrive - Centrum Łukasiewicz/Pulpit/pix2pix/train/train'
valid_dir = 'C:/Users/kordian.czyzewski/OneDrive - Centrum Łukasiewicz/Pulpit/pix2pix/val/val'

train_list = os.listdir(train_dir)
valid_list = os.listdir(valid_dir)

path = train_dir + '/' + train_list[0] 

im = Image.open(path)

arr = np.array(im)

sat_arr = arr[:, :600, :]
map_arr = arr[:, 600:, :]

img_map = Image.fromarray(map_arr)
img_sat = Image.fromarray(sat_arr)

img_map.save('ex_map.png')
img_sat.save('ex_sat.png')

os.mkdir('train_img')
os.mkdir('train_img/map')
os.mkdir('train_img/sat')


for img in train_list:
    
    path = train_dir + '/' + img
    
    im = Image.open(path)
    
    arr = np.array(im)
    
    sat_arr = arr[:, :600, :]
    map_arr = arr[:, 600:, :]
    
    img_map = Image.fromarray(map_arr)
    img_sat = Image.fromarray(sat_arr)

    img_map.save(f'train_img/map/{img}.png')
    img_sat.save(f'train_img/sat/{img}.png')