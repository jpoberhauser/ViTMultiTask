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

        # we need one norm for encoder output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, ):
        return x


def get_spatial_position_embedding(pos_emb_dim, feat_map):
    r"""
    this is the 2D sinusoidal positional embedding that lets the 
    transformer know where each patch of the grid cell in the image comes from. 
    """
    assert pos_emb_dim % 4 == 0, ('Position embedding dimension '
                                  'must be divisible by 4')
    grid_size_h, grid_size_w = feat_map.shape[2], feat_map.shape[3]
    grid_h = torch.arange(grid_size_h,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid_w = torch.arange(grid_size_w,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Number of grid cell tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=feat_map.device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([
        torch.sin(grid_h_emb),
        torch.cos(grid_h_emb)
    ], dim=-1)
    # grid_h_emb -> (Number of grid cell tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([
        torch.sin(grid_w_emb),
        torch.cos(grid_w_emb)
    ], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of grid cell tokens, pos_emb_dim)
    return pos_emb

class TransformerDecoderLayer(nn.Module):
    r"""
            Decoder for transformer of DETR, sequence of decoder layers. 
            Each layer has:
            1. LAyer norms for self-attn
            2. self-attn
            3. Layer norm for cross-attn
            4. cross-attn
            5. Layer norm for MLP
            6. MLP
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim, dropout_prob = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        ## self-attention for decoder
        self.attns = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                      batch_first=True)
               for _ in range(num_layers)                       
        ])

        # cross-attn
        # this gives the queries access to the global image features
        self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads, dropout=dropout_prob,
                                      batch_first=True)
               for _ in range(num_layers)                       
        ])

        ## MLP
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

        self.cross_attn_norms = nn.ModuleList([
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

        self.cross_attn_dropouts = nn.ModuleList([
                nn.Dropout(dropout_prob)
                for _ in range(num_layers)
        ])

        ## dropoutes for MLP 
        self.ff_dropouts = nn.ModuleList([
                nn.Dropout(dropout_prob)
                for _ in range(num_layers)
        ])

        # shared output norm for all decoder outputs
        # we will use this to norm ouput of each layer before senfing to box and class mLP
        # remeber, the output of each layer in decoder goes mto box and class MLP for loss calc.
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_ouput, query_embedding, spatial_positional_encoding):
        r"""
        query_objects: (batch_size, num_queries, d_model) these are the object queries
        encoder_output: (batch_size, seq_len, d_model) these are the image features from encoder
        query_embedding: (num_queries, d_model) these are the learnable embeddings for object queries
        spatial_positional_encoding: (batch_size, seq_len, d_model) these are the positional encodings for image features
        """
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []

        return x



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

        self.encoder = TransformerEncoderLayer(num_layers=config['num_encoder_layers'],
                                               num_heads=config['num_heads'],
                                               d_model=self.d_model,
                                               ff_inner_dim=config['ff_inner_dim'],
                                               dropout_prob=config['dropout_prob'])
        # these are the object queries, creates learnable embeddings
        # this serves as the input query sequence to the decoder
        # it is the decoder's job to take these and transforme them into meaningful object representations
        # why nn.Parameter and not nn.Embedding? no need for lookup table functionality
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.d_model))
        # Decoder
        self.decoder = TransformerDecoderLayer(num_layers = config['decoder_layers'],
                                                num_heads = config['decoder_attn_heads'],
                                                d_model = config['d_model'],
                                                ff_inner_dim=config['ff_inner_dim'], 
                                                dropout_prob=config['dropout_prob'])
        # classification mlp
        self.class_mlp = nn.Linear(self.d_model, self.num_classes)
        # bbox mlp
        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 4) # one per cx, cy, w, h
        )


        ## Placeholder for keypoint head, 2 MLPs, one for class and one for kpt coords
        # need to figure out how number of queries and num_kpts relate
        self.class_mlp_kpt = nn.Linear(self.d_model, self.num_kpts)
        self.kpt_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.num_kpts * 2) # one per x, y
        )

    def forward(self, x, targets = None, score_thresh = 0, use_nms = False):
        # x -> [B, C, H, W]
        # default d_model is 256
        # default C is 512 for resnet34
        # default H,X is 640x640
        # default feat_h, feat_w - 20, 20 stride factor is 32

        # CNN backbone
        conv_out = self.backbone(x) # [B, C_backbone,feat_h, feat_w] the default c channles of backbone is 512
        # project
        conv_out = self.backbone_proj(conv_out) # [B, d_model, feat_h, feat_w] from 512 to d_model (256)
        batch_size, d_model, feat_h, feat_w = conv_out.shape

        spatial_pos_embed = get_spatial_position_embedding(self.d_model, conv_out) #512x256
        # spatia positional emebddings

        # image features are still a gird, we need to convert to a sequence before it goes to transformer
        conv_out = (conv_out.reshape(batch_size, d_model, feat_h * feat_w).transpose(1,2))
        encode_out = self.encoder(conv_out)