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

    def forward(self, x, spatial_positional_embedding):
        out = x
        attn_weights = []
        for layer_idx in range(self.num_layers): # go through each layer
            # norm, mha, 
            in_attn = self.attn_norms[layer_idx](out)
            # add spatial positional embedding to in_attn
            q = in_attn + spatial_positional_embedding
            k = in_attn + spatial_positional_embedding
            out_attn, attn_w = self.attns[layer_idx](query = q,
                                                     key =  k,
                                                      value=  in_attn)
            attn_weights.append(attn_w)
            out_attn = self.attn_dropouts[layer_idx](out_attn)
            out = out + out_attn # residual connection

            # norm, mlp, residual
            in_ff = self.ff_norms[layer_idx](out)
            out_ff = self.ffs[layer_idx](in_ff)
            out_ff = self.ff_dropouts[layer_idx](out_ff)
            out = out + out_ff # residual connection

        # one last output norm, attn weights is just for viz
        out = self.output_norm(out)
        return out, torch.stack(attn_weights, dim=1) # (batch_size, num_layers, num_heads, seq_len, seq_len)


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

    def forward(self, query_objects, encoder_ouput, query_embedding, 
                spatial_positional_encoding):
        r"""
        query_objects: (batch_size, num_queries, d_model) these are the object queries
        encoder_output: (batch_size, seq_len, d_model) these are the image features from encoder
        query_embedding: (num_queries, d_model) these are the learnable embeddings for object queries
        spatial_positional_encoding: (batch_size, seq_len, d_model) these are the positional encodings for image features
        """
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []
        for i in range(self.num_layers):
            # norm, mha, res, 
            in_attn = self.attn_norms[i](out)
            q = in_attn + query_embedding
            k = in_attn + query_embedding
            out_attn, _ = self.attns[i](query = q,
                                        key = k,
                                        value = in_attn)
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn # residual connection

            # norm, cross-attn, dropout, res
            in_cross_attn = self.cross_attn_norms[i](out)
            q = in_cross_attn + query_embedding
            k = encoder_ouput + spatial_positional_encoding # 3cnoder features with spatial position
            out_cross_attn, cross_attn_w = self.cross_attn[i](query = q,
                                                               key = k,
                                                               value = encoder_ouput)
            
            decoder_cross_attn_weights.append(cross_attn_w)
            out_cross_attn = self.cross_attn_dropouts[i](out_cross_attn)
            out = out + out_cross_attn # residual connection

            
            # norm, mlps, droppit, res
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff # residual connection
            decoder_outputs.append(self.output_norm(out)) # decoder returns these

        output = torch.stack(decoder_outputs, dim=1) # (batch_size, num_layers, num_queries, d_model)
        return output, torch.stack(decoder_cross_attn_weights, dim=1) # (batch_size, num_layers, num_heads, num_queries, seq_len)



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
        enc_output, enc_attn_weights = self.encoder(conv_out, spatial_pos_embed) # <- pass in outs from conv and spatial positional embeddings
        # encode_out -> [B, feat_h*feat_w, d_model]
        query_objects = torch.zeros_like(self.query_embed.unsqueeze(0).repeat(batch_size,1,1)) # all zero tensor
        # query_objects -> [B, num_queries, d_model] these are the object queries
        query_objects, deco_attn_weights = self.decoder(query_objects,
                                    enc_output,
                                    self.query_embed.unsqueeze(0).repeat(batch_size,1,1),#actual embeddings for the queries
                                    spatial_pos_embed)
        #query_ojects = [num_decoder_layers, B, num_queries,d_model]
        # decoder takes in the encoder ouput _and_ the object queries

        cls_output = self.class_mlp(query_objects)
        # (num_decoder_layers, B, num_queries, num_classes)
        bbox_output = self.bbox_mlp(query_objects).sigmoid() # sigmoid to keep in [0,1]
         # (num_decoder_layers, B, num_queries, 4)


        losses = defaultdict(list)
        detections = []
        detr_output = {}

        if self.training:
            num_decoder_layers = self.num_decoder_layers
            # permofrm matching for each decoder layer
            # ----   matching --------
            for decoder_idx in range(num_decoder_layers): # aux loss <- this happens every single decoder layer!!
                cls_idx_output = cls_output[decoder_idx]
                bbox_idx_output = bbox_output[decoder_idx]
                with torch.no_grad(): # mathcin ghappens without gradients! Not smooth. 
                    # concat all prediction boxes and classes across batch
                    class_prob = cls_idx_output.reshape(-1, self.num_classes).softmax(-1)
                    # class_prob -> (B* num_queries, num_classes)
                    pred_boxes = bbox_idx_output.reshape(-1,4)
                    # pred_boxes -> (B* num_queries, 4)

                    # concat all target boxes and classes across batch
                    target_labels = torch.cat([t['labels'] for t in targets])
                    target_boxes = torch.cat([t['boxes'] for t in targets])

                    # Get cost, for classification its the  1 - prob
                    cost_classification = -class_prob[:, target_labels] # (B* num_queries, total_num_target_boxes)

                    # DETR predicts cx, cy, w, h, we need to convert to x1yx2y2 for giou
                    # giou cost
                    pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(pred_boxes,
                                                                      in_fmt = 'cxcywh',
                                                                      out_fmt = 'xyxy')
                    cost_localization_l1 = torch.cdist(pred_boxes_x1y1x2y2, target_boxes, p=1) # (B* num_queries, total_num_target_boxes)
                    cost_localization_giou = -torchvision.ops.generalized_box_iou(pred_boxes_x1y1x2y2,
                                                                                 target_boxes) # (B* num_queries, total_num_target_boxes)
                    total_cost = (self.l1_cost_weight * cost_localization_l1 +
                                  self.giou_cost_weight * cost_localization_giou +
                                  self.cls_cost_weight * cost_classification)
                    # aggregated cost matrix
                    total_cost = total_cost.reshape(batch_size, self.num_queries, -1).cpu()

                    num_targets_per_image = [len(t['labels']) for t in targets]
                    total_cost_per_batch_image = total_cost.split(num_targets_per_image, dim=-1)
                    # now we match
                    match_indices = [] # hold list of assignements per batch
                    for batch_idx in range(batch_size):
                        batch_idx_pred, batch_idx_target = linear_sum_assignment( #< hungarian algo
                            total_cost_per_batch_image[batch_idx][batch_idx])
                        match_indices.append((torch.as_tensor(batch_idx_pred, dtype=torch.int64),
                                              torch.as_tensor(batch_idx_target, dtype=torch.int64)))
                        # match indeces -> 2 sequnces, list of matched predictions and the second is amtched target box indexes. 
                        # first element of both are matched to each other, the second element of both are matched to each other, etc.
                        # only has prediction box indeces that are matched. We need to figure out what to do with unmatched ones later <- (unassigned)
                        # ([pred_box_a1,...], t[target_box_a1,,...]),
                        # ([pred_box_a2, ], [t[target_box_a2,...]]),
                        # has the assignment pairs for ith batch image. 

                # ------- end matching --------
                pred_batch_idxs = torch.cat([
                    torch.ones_like(pred_idx) * i
                    for i, (pred_idx, _) in enumerate(match_indices)
                ])
                pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])

                valid_obj_target_cls = torch.cat([
                    target['labels'][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)
                ])# get sequence of target box indeces, get the lables for gt boxes at those indeces

                target_classes = torch.full(
                    clas_idx_output.shape[:2],
                    fill_value = self.bg_class_idx, # initially everything is set as background
                    dtype = torch.int64,
                    device = cls_idx_output.device
                )
                # for predicted boxes that are assigned to a tagert in hungarian algo
                # we update those classes from background to target label
                # everything else stays as background
                target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls

                # We need to ensure background class is not deisproportinoately atteneded to in the model,
                # so we weight it
                cls_weights = torch.ones(self.num_classes)
                cls_weights[self.bg_class_idx] = self.bg_class_idx #(0.1) is default

                # now we're ready for classification loss
                loss_cls = torch.nn.functional.cross_entropy(
                    cls_idx_output.reshape(-1, self.num_classes),
                    target_classes.reshape(-1),
                    cls_weights.to(cls_idx_output)
                )

                # now we need localization loss, 2 parts
                matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]
                # we only care about matched boxes
                # get targets for those
                target_boxes = torch.cat([
                    target['boxes'][target_obj_idx]
                           for target, (_, target_obj_idx) in zip(targets, match_indices)],
                           dim=0
                )

                matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                    matched_pred_boxes, in_fmt = 'cxcywh',
                    out_fmt = 'xyxy'
                )
                # L1 Loss
                loss_l1_box = torch.nn.functional.l1_loss(
                    matched_pred_boxes_x1y1x2y2,
                    target_boxes,
                    reduction = 'none'
                )
                loss_l1_box = loss_l1_box.sum() / matched_pred_boxes.shape[0]

                # GIoU loss
                loss_giou = torchvision.ops.generalized_box_iou_loss(
                    matched_pred_boxes_x1y1x2y2,
                    target_boxes,
                )

                loss_giou = loss_giou.sum() / matched_pred_boxes.shape[0]

                losses['classificatoin'].append(loss_cls * self.cls_cost_weight)
                losses['bbox_regression'].append(
                    loss_l1_box * self.l1_cost_weight +
                    loss_giou * self.giou_cost_weight
                )

            detr_output['loss'] = losses # need to update for every layer, and thenn we'll use for backward pass

        else:
            #inference code (no backprop)
            # no matching and no loss nceessary, no need to get intere=mediate layer outputs
            cls_output = cls_output[-1] #last layer outs
            bbox_output = bbox_output[-1]# last layer outs

            prob = torch.nn.functional.softmax(cls_output, -1) # confs

            # get all query boxes and their best foreground class as label
            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels = labels + 1
            else:
                scores, labels =  prob[..., 1:].max(-1)

            #convert back to x1y1x2y2
            boxes = torchvision.ops.box_convert(bbox_output,
                                                in_fmt = 'cxcywh',
                                                out_fmt = 'xyxy')
            
            for batch_idx in range(boxes.shape[0]):
                scores_idx = scores[batch_idx]
                labels_idx = labels[batch_idx]
                boxes_idx = boxes[batch_idx]

            detr_output['detections'] = detections
            detr_output['enc_attn'] = enc_attn_weights # for vis
            detr_output['dec_attn'] = deco_attn_weights# for vis
        return detr_output

