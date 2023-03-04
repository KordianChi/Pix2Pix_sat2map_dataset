from mapsat_dataset import SatMapDataset
from sat2map_model import Discriminator
from sat2map_model import Generator

from torch import device
from torch import abs as ab
from torch import cat
from torch import ones
from torch import float as flt
from torch import zeros
from torch import save
from torchvision.transforms import transforms
from torch.cuda import is_available
from torch.utils.data import DataLoader

from torch.nn import BCELoss
from torch.optim import Adam

import os
import time

TRAIN_SAT_DIR = 'train_img/sat'

TRAIN_MAP_DIR = 'train_img/map'

BATCH_SIZE = 100

EPOCHS = 2

lr = 0.0002
beta1 = 0.5
beta2 = 0.999
l1_lambda = 100

devices = device("cuda:0" if is_available() else "cpu")

transform_map = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((128, 128)),
                                    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                         std=[1.0, 1.0, 1.0]),
                                   ])


transform_sat = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((128, 128)),
                                    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                         std=[1.0, 1.0, 1.0]),
                                   ])

img_list = os.listdir('train_img/sat')

train_data = SatMapDataset(img_list, TRAIN_MAP_DIR, TRAIN_SAT_DIR,
                           transform_sat, transform_map)


train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

discriminator = Discriminator().to(devices)

generator = Generator().to(devices)

criterion = BCELoss()

disc_optim = Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
gen_optim = Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

for _ in range(EPOCHS):
    
    start = time.time()
    
    for step, data in enumerate(train_dataloader):
        
        start_step = time.time()
        
        sat, img = data["final_sat"].to(devices), data["final_map"].to(devices)
        
        # first disc train step
        
        discriminator.zero_grad()
        
        real_data = cat([sat, img], dim=1).to(devices)
        outputs = discriminator(real_data)
        labels = ones(size=outputs.shape, dtype=flt, device=devices)
        
        disc_loss_real = criterion(outputs, labels)
        disc_loss_real.backward()
        
        # second disc train step
        
        gen = generator(sat).detach()
        
        fake_data = cat((sat, gen), dim=1)
        
        outputs = discriminator(fake_data)
        
        labels = zeros(size=outputs.shape, dtype=flt, device=devices)
        
        disc_loss_fake = criterion(outputs, labels)
        disc_loss_fake.backward()
        
        disc_optim.step()
        
        for i in range(2):
            
            generator.zero_grad()
            
            gen = generator(sat)
            
            gen_data = cat([sat, gen], dim=1)
            outputs = discriminator(gen_data)
            labels = ones(size = outputs.shape, dtype=flt, device=devices)
            
            gen_loss = criterion(outputs, labels) + l1_lambda * ab(gen - img).sum()
            gen_loss.backward()
            gen_optim.step()
            
        end_step = time.time()
        print(end_step - start_step)
        
    end = time.time()
    print(end - start)
save(generator, 'model_pix2pix_5')
print('MODEL SAVED')    
                