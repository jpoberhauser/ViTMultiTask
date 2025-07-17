# MultiTaskViT


Leveraging the power of ViT and SSL for a multi-task vision model that can perform pose estimation, object detection, and instance segmentation on a shared backbone. 

## Source Models

### Segmentation

[EoMT](https://huggingface.co/docs/transformers/main/model_doc/eomt). Encoder-only Mask Transformer (CVPR 2025 highlight)


### Object Detection


[YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos). Leverages plain ViT for object detection. 

### Pose/Keypoints

[TokenPose](https://github.com/leeyegy/TokenPose)

	•	Append N learnable keypoint tokens (one per joint)

	•	Feed them through the shared ViT encoder
	
    •	Decode their spatial predictions using MLP heads or 1x1 convs over feature maps (via spatial map reconstruction or regression)

[VitPose](https://arxiv.org/abs/2204.12484)

* "It employs a standard, non-hierarchical ViT backbone and a simple decoder head to predict keypoint heatmaps from images. Despite its simplicity, ViTPose achieves top results on the MS COCO Keypoint Detection benchmark."
