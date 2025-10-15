## Implementing DETR from original paper

* then we replace with LW-DETR which replaces the resnet 50 backbone with a pure ViT to make it more lightweight. 



# DETR Architecture: High-Level Diagram

## Input & Feature Extraction
+-------------------+
|     Image Input   |  <-- x_img ∈ R^(3 x H0 x W0) [
+-------------------+
          |
          v
+-------------------+
|   CNN Backbone    |  <-- Conventional CNN (e.g., ResNet ) 
| (Feature Extractor) |
+-------------------+
          |
          v
+-------------------+
|  2D Image Features|  <-- Feature Map f ∈ R^(C x H x W)  
+-------------------+

## Transformer Encoder (Global Context Reasoning)
          |
          |  (1x1 Conv reduces channels, flatten spatial dims) 
          v
+-------------------+
|  Encoder Input    |  <-- Flattened Features + Spatial Positional Encoding
+-------------------+
          |
          v
+-------------------+
| Transformer Encoder |  <-- Uses Multi-Head Self-Attention for global context
+-------------------+
          |
          v
+-------------------+
|  Encoder Output   |  <-- Contextualized Image Features (Memory for Decoder) 
+-------------------+

## Transformer Decoder (Parallel Prediction)
+------------------------------------------------------------------------------------------------------------------+
| Inputs to Decoder:                                                                                               |
| 1. Encoder Output (Memory) [9]                                                                                  |
| 2. Learned Object Queries (Input Positional Embeddings, fixed small set N, e.g., N=100)        |
+------------------------------------------------------------------------------------------------------------------+
          |
          v
+---------------------+
| Transformer Decoder |  <-- Decodes N objects in parallel using Self- & Cross-Attention 
+---------------------+
          |
          v
+------------------+------------------+--- ... ---+------------------+
| Decoder Output 1 | Decoder Output 2 |   ...   | Decoder Output N |
+------------------+------------------+--- ... ---+------------------+
          |                  |                       |
          v                  v                       v
+--------+---------+ +--------+---------+       +--------+---------+
|     FFN (Class)  | |     FFN (Class)  |       |     FFN (Class)  |  <-- Shared Feed Forward Networks 
|      FFN (Box)   | |      FFN (Box)   |       |      FFN (Box)   |
+--------+---------+ +--------+---------+       +--------+---------+
          |                  |                       |
          v                  v                       v
+------------------+------------------+--- ... ---+------------------+
| Final Prediction 1| Final Prediction 2|   ...   | Final Prediction N |  <-- Class + Bounding Box, or "No Object" (∅) 
+------------------+------------------+--- ... ---+------------------+
(Trained End-to-End with Bipartite Matching Loss to enforce unique predictions) [cx, cy, w, h]


## Main Recap on cost and loss

* **cost** of assignment is class (prob(target)) + localization cost. (L1 + GIoU) --> Hungarian matching. 

* boxes that match are foreground, the unassigned boxes are background. Model is trained then with classification **loss** (cross_entropy) and localization loss (smooth_l1 and GIoU Loss) 

    * decoder layer outputs go into **auxiliary losses**. They all use the same shared MLP, and ten total loss is a sum of losses across all layers. This is for training only. 

  