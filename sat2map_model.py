from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import LeakyReLU
from torch.nn import BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn import Dropout
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import Sigmoid
from torch import cat



### --- DISCRIMINATOR MODEL --- ###


class Discriminator(Module):
    
    '''
    Markovian Discirminator model
    '''
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = Sequential(
            
            # convolution k=4, s=2, p=1 halved size
            
            Conv2d(6, 64, 4, 2, 1, bias=False),
            BatchNorm2d(64),
            LeakyReLU(0.2),
            
            Conv2d(64, 128, 4, 2, 1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(0.2),
            
            Conv2d(128, 256, 4, 2, 1, bias=False),
            BatchNorm2d(256),
            LeakyReLU(0.2),
            
            # convolution k=3, s=1, p=1 size invariant
            
            Conv2d(256, 512, 3, 1, 1, bias=False),
            BatchNorm2d(512),
            LeakyReLU(0.2),
            
            Conv2d(512, 1, 3, 1, 1, bias=False),
            Flatten(),
            Linear(256, 1),
            Sigmoid()
            )
        
    def forward(self, inputs):
        
        return self.main(inputs)


### --- GENERATOR MODEL ---###


def conv2d_block(channel_in, channel_out):
    
    # convolution k=4, s=2, p=1 halved size
    
    block = Sequential(
        Conv2d(channel_in, channel_out, 4, 2, 1, bias=False),
        BatchNorm2d(channel_out),
        LeakyReLU(0.2),
        )
    
    return block
    
    
def  deconv2d_block(channel_in, channel_out):
    
    # deconvolution k=4, s=2, p=1 doubled size
    
    block = Sequential(
        ConvTranspose2d(channel_in, channel_out, 4, 2, 1, bias=False), 
        BatchNorm2d(channel_out),
        Dropout(0.5),
        ReLU()
        )
    
    return block

class Generator(Module):
    
    '''
    Based on U-Net style architecture
    '''
    
    def __init__(self):
        super(Generator, self).__init__()
    
        ### Encoder part
    
        self.encoder_input = Conv2d(3, 64, 4, 2, 1, bias=False)
        
        self.encoder_block_1 = conv2d_block(64, 128)
        self.encoder_block_2 = conv2d_block(128, 256)
        self.encoder_block_3 = conv2d_block(256, 512)
        self.encoder_block_4 = conv2d_block(512, 512)
        self.encoder_block_5 = conv2d_block(512, 512)
        
        self.encoder_output = Conv2d(512, 512, 4, 2, 1, bias=False)
    
        ### Decoder part
        
        self.decoder_input = ReLU()
        
        self.decoder_block_1 = deconv2d_block(512, 512)
        self.decoder_block_2 = deconv2d_block(1024, 512)
        self.decoder_block_3 = deconv2d_block(1024, 512)
        self.decoder_block_4 = deconv2d_block(1024, 256)
        self.decoder_block_5 = deconv2d_block(512, 128)
        self.decoder_block_6 = deconv2d_block(256, 64)
        
        self.decoder_output = ConvTranspose2d(128, 3, 4, 2, 1, bias=False)
        
        self.generator_output = Tanh()
        
    def forward(self, inputs):

        ### Encoder step        

        enc_1 = self.encoder_input(inputs)
        
        enc_2 = self.encoder_block_1(enc_1)
        enc_3 = self.encoder_block_2(enc_2)
        enc_4 = self.encoder_block_3(enc_3)
        enc_5 = self.encoder_block_4(enc_4)
        enc_6 = self.encoder_block_5(enc_5)
        
        enc_out = self.encoder_output(enc_6)
        
        ### Decoder step
        
        dec_inp = self.decoder_input(enc_out)
        
        dec_1 = self.decoder_block_1(dec_inp)
        dec_2 = cat((dec_1, enc_6), dim=1)
        
        dec_3 = self.decoder_block_2(dec_2)
        dec_4 = cat((dec_3, enc_5), dim=1)
        
        dec_5 = self.decoder_block_3(dec_4)
        dec_6 = cat((dec_5, enc_4), dim=1)
        
        dec_7 = self.decoder_block_4(dec_6)
        dec_8 = cat((dec_7, enc_3), dim=1)
        
        dec_9 = self.decoder_block_5(dec_8)
        dec_10 = cat((dec_9, enc_2), dim=1)
        
        dec_11 = self.decoder_block_6(dec_10)
        dec_12 = cat((dec_11, enc_1), dim=1)
        
        dec_out = self.decoder_output(dec_12)
        
        gen_out = self.generator_output(dec_out)
        
        return gen_out
        