from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class SatMapDataset(Dataset):
    
    def __init__(self, img_list, map_dir, sat_dir, transform_sat=None, 
                transform_map=None, train=True):
        
        super(SatMapDataset, self).__init__()
        self.img_list = img_list
        self.map_dir = map_dir
        self.sat_dir = sat_dir
        self.transform_map = transform_map
        self.transform_sat = transform_sat
        self.train = train
        
        
    def __len__(self):
        
        return(len(self.img_list))
    
    def __getitem__(self, index):
        
        map_path = os.path.join(self.map_dir, self.img_list[index])
        sat_path = os.path.join(self.sat_dir, self.img_list[index])
        
        
        arr_map = np.array(Image.open(map_path))
        arr_sat = np.array(Image.open(sat_path))
        
        if self.transform_map:
            
            final_map = self.transform_map(arr_map)
            
        if self.transform_sat:
        
            final_sat = self.transform_sat(arr_sat)
            
        return {'final_sat': final_sat, 'final_map': final_map}
            
            