# Description: Contains functions for decomposing a given model
# Author: XXXXXXXXXXXX
# Last Modified: 2023/11/20


import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import logging
import torch.nn.utils.weight_norm as weight_norm


logger = logging.getLogger(__name__)

def decompose_and_replace_layer_by_name(module, layer_name, rank=None, freeze=False, device='cpu', only_replace=False):    # change name later
    
    if rank is None:
        raise ValueError("Please specify a rank for decomposition")
    
    rank = torch.tensor(rank, dtype=torch.int32)
    if device=='cuda':
        rank=rank.to('cuda' if device=='cuda' else 'cpu')
    
    found = False
    error = None
    queue = [(name,layer,module,name) for name, layer in list(module.named_children())]
    while queue:
        (name,layer,parent,fullname) = queue.pop()
        if isinstance(layer,nn.Conv2d):
            if layer_name == fullname:
                logger.warning(f"Convolution layer decomposition is not supported yet. Skipping...")
                return [None], 0, -1

        elif isinstance(layer,nn.Linear) and layer_name == fullname:
            found = True
            new_layers, error, layer_compress_ratio, R = svd_decomposition_fc_layer(layer, rank, only_replace=only_replace)
            new_layers = new_layers.to(device)
            setattr(parent, name, new_layers)
            break
        
        children = list(layer.named_children())
        if len(children)>0:
            queue.extend([(name,child,layer,fullname+'.'+name) for name,child in children])
    if freeze:
        for name, param in new_layers.named_parameters():
            param.requires_grad = False
    if not found:
        return [None], 0, -1
    return  error, layer_compress_ratio, R

def svd_decomposition_fc_layer(layer, rank, only_replace=False):

    layer_total_params = sum(p.numel() for p in layer.parameters()) ##newline
    if not only_replace:
        cont = True
        while cont:
            U,S,Vt = torch.linalg.svd(layer.weight.data, full_matrices=False)
            logger.info('U shape: {}'.format(U.shape))
            logger.info('S shape: {}'.format(S.shape))
            logger.info('Vt shape: {}'.format(Vt.shape))

            logger.info('ATTENTION: sum of lower singular values are: {}'.format(torch.sum(torch.log(S[rank:]), dim=0).item()))
            logger.info((torch.log(S[rank:])<0).any().item())
            logger.info(torch.log(S[rank:]))

            U = U[:,:rank]
            S = S[:rank]
            Vt = Vt[:rank,:]
            

            logger.info('U shape: {}'.format(U.shape))
            logger.info('S shape: {}'.format(S.shape))
            logger.info('Vt shape: {}'.format(Vt.shape))

            S_sq = torch.diag(torch.sqrt(S))
            U2 = torch.mm(U,S_sq) # size must be (out_features, rank)
            logger.info('U2 shape: {}'.format(U2.shape))
            Vt2 = torch.mm(S_sq,Vt) # size must be (rank, in_features)
            logger.info('Vt2 shape: {}'.format(Vt2.shape))
            decomp_err = torch.norm(torch.mm(U2,Vt2)-layer.weight.data)
            logger.info('Decomposition error: {}'.format(decomp_err))
            logger.info('Factor norms: {}, {}'.format(torch.norm(U2), torch.norm(Vt2)))
            
            c_out, c_in = U2, Vt2
            if torch.isnan(c_out).any() or torch.isnan(c_in).any():
                logger.info(f"NaN detected in CP decomposition, trying again with rank {int(rank/2)}")
                rank = int(rank/2)
            else:
                cont = False
    else:
        (_,factors) = random_cp(layer.weight.data.shape, rank=rank)
        c_out, c_in = factors[0], factors[1]
        decomp_err = None

    bias_flag = layer.bias is not None

    fc_1 = torch.nn.Linear(in_features=c_in.shape[1], \
            out_features=rank, bias=False)

    fc_2 = torch.nn.Linear(in_features=rank, 
            out_features=c_out.shape[0], bias=bias_flag)

    if bias_flag:
        fc_2.bias.data = layer.bias.data
    
    fc_1.weight.data = c_in #torch.transpose(c_in,1,0)
    fc_2.weight.data = c_out

    new_layers = nn.Sequential(fc_1, fc_2)
    logger.info('new layers: {}'.format(new_layers))

    layer_compressed_params= sum(p.numel() for p in new_layers.parameters()) ##newline
    layer_compress_ratio = ((layer_total_params-layer_compressed_params)/layer_total_params)*100 ##newline

    return new_layers, decomp_err, layer_compress_ratio, rank    

# Calculates the cp-rank for a given layer to achieve a given compression ratio in layer-wise manner
def calculate_rank_for_ratio(layer, ratio):
    shape = torch.tensor(layer.weight.shape)
    R = (ratio*torch.prod(shape).item()) / torch.sum(shape).item()
    return max(1, int(R))

# Given a layer-name, calculates and returns the cp-rank for that layer to achieve a given compression ratio
def get_rank_for_layer_by_name(model, layer_name, ratio):
    #found = False
    for name, l in model.named_modules():
        if name == layer_name:
            if  hasattr(l, 'weight'):
                #found = True
                return calculate_rank_for_ratio(l, ratio)
            #else:
                #logger.info(f'layer {layer_name} is not a valid layer')
    return -1
    #raise ValueError(f'layer {layer_name} not found in model')

# Given a model, calculates and returns the cp-rank for all layers to achieve a given compression ratio per layer
def get_ranks_per_layer(model, ratio, layers_):
    layers = []
    ranks = []
    for name in layers_:
        R = get_rank_for_layer_by_name(model, name, ratio)
        if R != -1:
            ranks.append(R)
            layers.append(name)
        else:
            logger.info(f'layer {name} or given ratio is not valid. Skipping...')
    return layers, ranks
       
'''
def decompose_model_by_file(file_path):
    f = torch.load(file_path)
    replaced_layers = f['replaced_layers']
    layers = f['layers']
    ranks = f['ranks']

    for l,r in zip(layers, ranks):
        if l in replaced
'''


class DecompositionInfo:
    def __init__(self):
        self.layers = []
        self.ranks = []
        self.reduction_ratio = []
        self.approx_error = []

    def append(self, layer, rank, approx_error, ratio=None):
        self.layers.append(layer)
        self.ranks.append(rank)
        self.approx_error.append(approx_error)
        if ratio is not None:
            self.reduction_ratio.append(ratio)
    def isNone(self):
        if len(self.layers) > 0:
            return False
        return True
        

class Compression:
    def __init__(self):
        self.decomposition_info = DecompositionInfo()

    def apply_decomposition_from_checkpoint(self, model, decomposition_info:DecompositionInfo):
        for layer, rank in zip(decomposition_info.layers, decomposition_info.ranks):
            self.apply_layer_compression(model, layer, rank)
        self.decomposition_info = decomposition_info
        model = model.to(device=get_device())

    def apply_layer_compression(self, model, layer, rank, only_replace=False, freeze=False):
        
        logger.info('Decomposing layer {} with rank {}'.format(layer, rank))
        approx_error, layer_compress_ratio, decomp_rank = decompose_and_replace_layer_by_name(model, layer, rank, freeze=freeze, only_replace=only_replace) # for now set freeze to False
        try:
            self.decomposition_info.append(layer=layer, rank=decomp_rank, approx_error=approx_error[-1], ratio=layer_compress_ratio)
            logger.info('Layer Approximation error: {}, Layer Reduction ratio: {}'.format(approx_error[-1], layer_compress_ratio))
        except:
            self.decomposition_info.append(layer=layer, rank=decomp_rank, approx_error=-1, ratio=layer_compress_ratio)
        

    def apply_compression(self, model, layers, ranks, only_replace=False, freeze=False):
        logger.info('Layers to be compressed: %s', layers)
        steps = len(layers)
        for step in range(steps):
            layer_name = layers[step]
            rank = ranks[step]
            self.apply_layer_compression(model, layer_name, rank, only_replace=only_replace, freeze=freeze)

        return self.decomposition_info