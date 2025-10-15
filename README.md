# MultiTaskViT

WiP

Leveraging the power of ViT and SSL for a multi-task vision model that can perform pose estimation, object detection, and instance segmentation on a shared backbone. 



## RF-DETR-Seg

This could be a great candidate for instance segmentation + detection. 

We could then add the detection queries into [ED-POSE](https://github.com/IDEA-Research/ED-Pose). This approach extends the detection queries, to add a sequential query to find keypoints. 


Other option could be [DETR-POSE](https://github.com/SebastianJanampa/DETRPose) which extends the set prediction problem to kpts. 


## Other options



### Segmentation plus Detection

* RF-DETR-Seg is a great candidate for this. 

### Integrating Keypoint Heads into a DETR Framework

In DETR, each object query in the decoder produces an output embedding that is fed into lightweight heads (MLPs) for different predictions.

By default, one 3-layer MLP outputs the bounding box (4 numbers) and another outputs the class scores. We can extend this design by adding another MLP head to each query’s output to predict keypoint coordinates. 

For example, if we want  animal pose keypoints, a query could output $(x_1, y_1, x_2, y_2, \dots, x_K, y_K)$ for $K$ joints of that animal. In practice, researchers have implemented this in various ways. The ED-Pose method described above is one clear example: it attaches keypoint box regression to each query, effectively treating each keypoint like a small localized detection anchored to the person ￼. The model learns both the human’s bounding box and the keypoints in one go, using the same set of queries and the DETR bipartite matching paradigm for assignment of predictions to ground truth people ￼. Because DETR’s matching cost can be extended to include keypoint errors (or one can match primarily on boxes/class and then supervise keypoints after matching), the training can still find the correct query for each ground truth instance.


### Other approaches 

### Segmentation

[EoMT](https://huggingface.co/docs/transformers/main/model_doc/eomt). Encoder-only Mask Transformer (CVPR 2025 highlight)


### Object Detection


[YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos). Leverages plain ViT for object detection. 


Essentially, the change from a pre-trained ViT to a YOLOS detector is embarrassingly simple: 

	* YOLOS replaces one [CLS] token for image classification in ViT with one hundred [DET] tokens for object detection.

 	* YOLOS replaces the image classification loss in ViT with the bipartite matching loss to perform object
		detection in a set prediction manner following Carion et al. [10], which can avoid re-interpreting the
		output sequences of ViT to 2D feature maps as well as prevent manually injecting heuristics and prior
		knowledge of object 2D spatial structure during label assignment

		* YOLOS wants to predict sents of objects (multiple boxes and classes) so it uses bipartite matching loss.(from DETR)

		* YOLOS lets the transformer lear spatial structure on its own from data. No anchors, no grid cells, no 2D priors. 


* bounding box regression heads are implemented by one MLP with separate parameters containing two hidden layers with
intermediate ReLU [41] non-linearity activation functions.

### Pose/Keypoints

[TokenPose](https://github.com/leeyegy/TokenPose)

	•	Append N learnable keypoint tokens (one per joint)

	•	Feed them through the shared ViT encoder
	
    •	Decode their spatial predictions using MLP heads or 1x1 convs over feature maps (via spatial map reconstruction or regression)

[VitPose](https://arxiv.org/abs/2204.12484)

* "It employs a standard, non-hierarchical ViT backbone and a simple decoder head to predict keypoint heatmaps from images. Despite its simplicity, ViTPose achieves top results on the MS COCO Keypoint Detection benchmark."

## Install

### Mac M1
conda env create -f vit-multitask-m1.yaml
conda activate vit-multitask-m1
python -m ipykernel install --user --name=vit-multitask-m1


### Ubuntu
conda env create -f vit-multitask-env.yaml
conda activate vit-multitask
python -m ipykernel install --user --name=vit-multitask
