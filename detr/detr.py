import torch
import torchvision
from torchvision.models import resnet34
from torch import nn


class TransformerEncoderLayer(nn.Module):
    r"""
    Enocder for transformer of DETR, sequence of encoder layers. 
    Each layer has 1.) LayerNorm for SA
    2.) self-attention
    3.) LayerNorm for MLP
    4.) MLP
    """
    def __init__(self,num_layers, num_heads, d_model, ff_inner_dim, dropout_prob = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        # why ModuleListand not nn.Sequential?
        # ModuleList objects can be accessed like a list, and you can control the forward flow
        # if we have 4 layers, then we have a list of 4 mha that we can access in forward
        self.attns = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                      batch_first=True)
               for _ in range(num_layers)                       
        ])

        # init MLPS <- these are he MLPs inside the attention block, not the ouput ones
        self.ffs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
            for _ in range(num_layers)
        ])

        # init the Norms
        self.attn_norms = nn.ModuleList([
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
        ])

        # init the Norms for MLP
        self.ff_norms = nn.ModuleList([
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
        ])

        ## dropoutes for attn 
        self.attn_dropouts = nn.ModuleList([
                nn.Dropout(dropout_prob)
                for _ in range(num_layers)
        ])

        ## dropoutes for MLP 
        self.ff_dropouts = nn.ModuleList([
                nn.Dropout(dropout_prob)
                for _ in range(num_layers)
        ])




class DETR(nn.Module):
    r"""
    DETR model calss overseeing entire model. Forward pass:
    1.) backbone of CNN, resnet34
    2.) backbone feature map projection. Needs to be d_model as input to transformer
    3.) Transformer Encoder
    4.) Transformer Decoder
    5.) Box and CLass MLP    
    """

    def __init__(self, config, num_classes, bg_class_idx):
        super().__init__()
        self.backbone_channels = config['backbone_channels']
        self.d_model = config['d_model'] # default 256
        self.num_queries = config['num_queries'] # coco has 100, for us lets use 25
        self.num_classes = num_classes #voc is 21 including background
        self.num_decoder_layers = config['num_decoder_layers'] # defautl is 4, official is 6
        self.cls_cost_weight = config['cls_cost_weight']
        self.l1_cost_weight = config['l1_cost_weight']
        self.giou_cost_weight = config['giou_cost_weight']
        self.bg_cls_weight = config['bg_class_weight']# the weight we give to the background class when calculating classification cost
        self.nms_thresh = config['nms_thresh'] # only used for visualization
        self.bg_class_idx = bg_class_idx # usually 0, index of background class
        valid_gb_idx = (self.bg_class_idx ==0 or
                        self.bg_class_idx == self.num_classes -1)
        assert valid_gb_idx, "Background class must be at index 0 or num_classes -1"

        # CNN backbone
        self.backbone = nn.Sequential(*list(resnet34(
            weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
            norm_layer = torchvision.ops.FrozenBatchNorm2d).children())[:-2])
        # this backbone outputs feature map of shape [batch_size, 512, H/32, W/32]
        
        if config['freeze_backbone']:# this is is you want to freeze the backbone or not
            for p in self.backbone.parameters():# in papaer, authors train with smaller lr
                p.requires_grad = False
        # this prokector takes the 512 channels from backbone to d_model. (256)
        self.backbone_proj = nn.Conv2d(self.backbone_channels, self.d_model, kernel_size=1)

        self.encoder = 
    