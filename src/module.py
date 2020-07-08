import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Function

FBANK_SIZE = 80

class VGGExtractor(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''
    def __init__(self,input_dim):
        super(VGGExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel,freq_dim,out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.init_dim, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.init_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2) # Half-time dimension
                            )

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13,(13 // 4)*self.hide_dim
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE,(FBANK_SIZE//4)*self.hide_dim
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1,2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0],feature.shape[1],self.out_dim)
        return feature,feat_len


class FreqVGGExtractor(nn.Module):
    ''' Frequency Modification VGG extractor for ASR '''
    def __init__(self,input_dim, split_freq, low_dim=4):
        super(FreqVGGExtractor, self).__init__()
        self.split_freq =    split_freq
        self.low_init_dim  = low_dim
        self.low_hide_dim  = low_dim * 2
        self.high_init_dim = 64 - low_dim
        self.high_hide_dim = 128 - low_dim * 2

        in_channel,freq_dim = self.check_dim(input_dim)
        self.in_channel     = in_channel
        self.freq_dim       = freq_dim
        self.low_out_dim    = split_freq // 4 * self.low_hide_dim
        self.high_out_dim   = (freq_dim - split_freq) // 4 * self.high_hide_dim
        self.out_dim        = self.low_out_dim + self.high_out_dim

        self.low_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_init_dim, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.low_init_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_hide_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2) # Half-time dimension
                            )
        self.high_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_init_dim, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.high_init_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_hide_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2) # Half-time dimension
                            )
        
        assert(self.split_freq % 4 == 0)
        assert(self.split_freq > 0 and self.split_freq < self.freq_dim)

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//4
        # crop sequence s.t. t%4==0
        if feature.shape[1]%4 != 0:
            feature = feature[:,:-(feature.shape[1]%4),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        low_feature = self.low_extractor(feature[:,:,:,:self.split_freq])
        high_feature = self.high_extractor(feature[:,:,:,self.split_freq:])
        # features : BS x 4 x T/4 x D/4 , BS x 124 x T/4 x D/4
        # BS x H x T/4 x D/4 -> BS x T/4 x H x D/4
        low_feature = low_feature.transpose(1,2)
        high_feature = high_feature.transpose(1,2)
        #  BS x T/4 x H x D/4 -> BS x T/4 x HD/4
        low_feature = low_feature.contiguous().view(low_feature.shape[0],low_feature.shape[1],self.low_out_dim)
        high_feature = high_feature.contiguous().view(high_feature.shape[0],high_feature.shape[1],self.high_out_dim)
        feature = torch.cat((low_feature, high_feature), dim=-1)
        return feature, feat_len

class VGGExtractor2(nn.Module):
    ''' VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf'''
    ''' Only downsample once '''
    def __init__(self,input_dim):
        super(VGGExtractor2, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel,freq_dim,out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.init_dim, self.init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.init_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2), stride=(1, 2)) # 
                            )

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13,(13 // 4)*self.hide_dim
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE,(FBANK_SIZE//4)*self.hide_dim
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//2
        # crop sequence s.t. t%4==0
        if feature.shape[1]%2 != 0:
            feature = feature[:,:-(feature.shape[1]%2),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/2xD/4 -> BSxT/2x128xD/4
        feature = feature.transpose(1,2)
        #  BS x T/2 x 128 x D/4 -> BS x T/2 x 32D
        feature = feature.contiguous().view(feature.shape[0],feature.shape[1],self.out_dim)
        return feature,feat_len

class FreqVGGExtractor2(nn.Module):
    ''' Frequency Modification VGG extractor for ASR '''
    def __init__(self,input_dim, split_freq, low_dim=4):
        super(FreqVGGExtractor2, self).__init__()
        self.split_freq =    split_freq
        self.low_init_dim  = low_dim
        self.low_hide_dim  = low_dim * 2
        self.high_init_dim = 64 - low_dim
        self.high_hide_dim = 128 - low_dim * 2
        # self.init_dim      =  64
        # self.low_hide_dim  =   8
        # self.high_hide_dim = 120

        in_channel,freq_dim = self.check_dim(input_dim)
        self.in_channel     = in_channel
        self.freq_dim       = freq_dim
        self.low_out_dim    = split_freq // 4 * self.low_hide_dim
        self.high_out_dim   = (freq_dim - split_freq) // 4 * self.high_hide_dim
        self.out_dim        = self.low_out_dim + self.high_out_dim

        # self.first_extractor = nn.Sequential(
        #                         nn.Conv2d( in_channel, self.init_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d( self.init_dim, self.init_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.MaxPool2d(2, stride=2), # Half-time dimension
        #                     )
        # self.low_extractor = nn.Sequential(
        #                         nn.Conv2d( self.init_dim, self.low_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d( self.low_hide_dim, self.low_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.MaxPool2d((1, 2), stride=(1, 2)) # 
        #                     )
        # self.high_extractor = nn.Sequential(
        #                         nn.Conv2d( self.init_dim, self.high_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d( self.high_hide_dim, self.high_hide_dim, 3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.MaxPool2d((1, 2), stride=(1, 2)) # 
        #                     )
        self.low_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_init_dim, self.low_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.low_init_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.low_hide_dim, self.low_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2), stride=(1, 2)) # 
                            )
        self.high_extractor = nn.Sequential(
                                nn.Conv2d( in_channel, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_init_dim, self.high_init_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2, stride=2), # Half-time dimension
                                nn.Conv2d( self.high_init_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d( self.high_hide_dim, self.high_hide_dim, 3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2), stride=(1, 2)) # 
                            )
        
        assert(self.split_freq % 4 == 0)
        assert(self.split_freq > 0 and self.split_freq < self.freq_dim)

    def check_dim(self,input_dim):
        # Check input dimension, delta feature should be stack over channel. 
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim // 13),13
        elif input_dim % FBANK_SIZE == 0:
            # Fbank feature
            return int(input_dim // FBANK_SIZE),FBANK_SIZE
        else:
            raise ValueError('Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+d)

    def view_input(self,feature,feat_len):
        # downsample time
        feat_len = feat_len//2
        # crop sequence s.t. t%4==0
        if feature.shape[1]%2 != 0:
            feature = feature[:,:-(feature.shape[1]%2),:].contiguous()
        bs,ts,ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs,ts,self.in_channel,self.freq_dim)
        feature = feature.transpose(1,2)

        return feature,feat_len

    def forward(self,feature,feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature,feat_len)
        # feature   = self.first_extractor(feature) # new 
        # Foward
        low_feature  = self.low_extractor(feature[:,:,:,:self.split_freq])
        high_feature = self.high_extractor(feature[:,:,:,self.split_freq:])
        # low_feature  = self.low_extractor(feature[:,:,:,:self.split_freq//2])
        # high_feature = self.high_extractor(feature[:,:,:,self.split_freq//2:])
        # features : BS x 4 x T/4 x D/4 , BS x 124 x T/4 x D/4
        # BS x H x T/4 x D/4 -> BS x T/4 x H x D/4
        low_feature = low_feature.transpose(1,2)
        high_feature = high_feature.transpose(1,2)
        #  BS x T/4 x H x D/4 -> BS x T/4 x HD/4
        low_feature = low_feature.contiguous().view(low_feature.shape[0],low_feature.shape[1],self.low_out_dim)
        high_feature = high_feature.contiguous().view(high_feature.shape[0],high_feature.shape[1],self.high_out_dim)
        feature = torch.cat((low_feature, high_feature), dim=-1)
        return feature, feat_len

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''
    def __init__(self, input_dim, module, dim, bidirection, dropout, layer_norm, sample_rate, sample_style, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2*dim if bidirection else dim
        self.out_dim = sample_rate*rnn_out_dim if sample_rate>1 and sample_style=='concat' else rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.sample_style = sample_style
        self.proj = proj

        if self.sample_style not in ['drop','concat']:
            raise ValueError('Unsupported Sample Style: '+self.sample_style)
        
        # Recurrent layer
        self.layer = getattr(nn,module.upper())(input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout>0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim,rnn_out_dim)

    def forward(self, input_x , x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()
        # ToDo: check time efficiency of pack/pad
        #input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output,_ = self.layer(input_x)
        #output,x_len = pad_packed_sequence(output,batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout>0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            batch_size,timestep,feature_dim = output.shape
            x_len = x_len//self.sample_rate

            if self.sample_style =='drop':
                # Drop the unselected timesteps
                output = output[:,::self.sample_rate,:].contiguous()
            else:
                # Drop the redundant frames and concat the rest according to sample rate
                if timestep%self.sample_rate != 0:
                    output = output[:,:-(timestep%self.sample_rate),:]
                output = output.contiguous().view(batch_size,int(timestep/self.sample_rate),feature_dim*self.sample_rate)

        if self.proj:
            output = torch.tanh(self.pj(output)) 

        return output,x_len


class BaseAttention(nn.Module):
    ''' Base module for attentions '''
    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self):
        pass

    def compute_mask(self,k,k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs,ts,_ = k.shape
        self.mask = np.zeros((bs,self.num_head,ts))
        for idx,sl in enumerate(k_len):
            self.mask[idx,:,sl:] = 1 # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(k_len.device, dtype=torch.bool).view(-1,ts)# BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn) # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(1) # BNxT x BNxTxD-> BNxD
        return output, attn


class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy,v)
        
        attn = attn.view(-1,self.num_head,ts) # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''
    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att  = None
        self.loc_conv = nn.Conv1d(num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim,bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh,ts,_ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs,self.num_head,ts)).to(k.device)
            for idx,sl in enumerate(self.k_len):
                self.prev_att[idx,:,:sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(self.prev_att).transpose(1,2))) # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(1,self.num_head,1,1).view(-1,ts,self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1) # BNx1xD
        
        # Compute energy and context
        energy = self.gen_energy(torch.tanh( k+q+loc_context )).squeeze(2) # BNxTxD -> BNxT
        output, attn = self._attend(energy,v)
        attn = attn.view(bs,self.num_head,ts) # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn
