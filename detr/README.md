## Implementing DETR from original paper

* then we replace with LW-DETR which replaces the resnet 50 backbone with a pure ViT to make it more lightweight. 


Contributions of the LW-DETR paper:


*  Encoder Backbone: LW-DETR uses a plain Vision Transformer (ViT) encoder instead of the conventional CNN backbone (like ResNet) typically used in DETR
  
*  adopts interleaved window and global attentions (a simple modification akin to ViTDet) within the ViT encoder, replacing some computationally costly global self-attention layers with window self-attention to reduce complexity


* LW-DETR uses a shallow DETR decoder with only 3 transformer decoder layers, significantly fewer than the 6 layers typically adopted by DETR and its variants, resulting in a measurable time reduction (e.g., from 1.4 ms to 0.7 ms for the tiny version)

## Main Recap on cost and loss

* **cost** of assignment is class (prob(target)) + localization cost. (L1 + GIoU) --> Hungarian matching. 

* boxes that match are foreground, the unassigned boxes are background. Model is trained then with classification **loss** (cross_entropy) and localization loss (smooth_l1 and GIoU Loss) 

    * decoder layer outputs go into **auxiliary losses**. They all use the same shared MLP, and ten total loss is a sum of losses across all layers. This is for training only. 

   