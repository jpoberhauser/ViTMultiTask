# MultiTaskViT


Leveraging the power of ViT and SSL for a multi-task vision model that can perform pose estimation, object detection, and instance segmentation on a shared backbone. 

## Source Models

### Segmentation

[EoMT](https://huggingface.co/docs/transformers/main/model_doc/eomt). Encoder-only Mask Transformer (CVPR 2025 highlight)


### Object Detection


[YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos). Leverages plain ViT for object detection. 

"""
Essentially, the change from a pre-trained ViT to a YOLOS detector is embarrassingly simple: 

	* YOLOS replaces one [CLS] token for image classification in ViT with one hundred [DET] tokens for object detection.

 	* YOLOS replaces the image classification loss in ViT with the bipartite matching loss to perform object
		detection in a set prediction manner following Carion et al. [10], which can avoid re-interpreting the
		output sequences of ViT to 2D feature maps as well as prevent manually injecting heuristics and prior
		knowledge of object 2D spatial structure during label assignment

		* YOLOS wants to predict sents of objects (multiple boxes and classes) so it uses bipartite matching loss.(from DETR)

		* YOLOS lets the transformer lear spatial structure on its own from data. No anchors, no grid cells, no 2D priors. 
"""

* bounding box regression heads are implemented by one MLP with separate parameters containing two hidden layers with
intermediate ReLU [41] non-linearity activation functions.

### Pose/Keypoints

[TokenPose](https://github.com/leeyegy/TokenPose)

	•	Append N learnable keypoint tokens (one per joint)

	•	Feed them through the shared ViT encoder
	
    •	Decode their spatial predictions using MLP heads or 1x1 convs over feature maps (via spatial map reconstruction or regression)

[VitPose](https://arxiv.org/abs/2204.12484)

* "It employs a standard, non-hierarchical ViT backbone and a simple decoder head to predict keypoint heatmaps from images. Despite its simplicity, ViTPose achieves top results on the MS COCO Keypoint Detection benchmark."
