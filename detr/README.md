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
(Trained End-to-End with Bipartite Matching Loss to enforce unique predictions) [1, 2, 12]
