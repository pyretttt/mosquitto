# Faster RCNN

Consists of two networks: RPN and FasterRNN. Additionally it uses ROI pooling head.

## RPN

Region proposal network is convolution network that works above some general purpose convolution feature extractor.
It slides over entire conv feature map and for each window position it computes two things:
- Probability of each anchor box to be there or not
- Position of anchor box (or displacement of anchor about center?)

### Anchors
***
Anchors are predefined rectangles of different aspect ratio and scale. Different scales allow model to be scale invariant.


### RPN Training

For each window position on convolution feature map there're WxHx{Anchors} predictions (binary label or softmax over all anchors?).
RPN can be trained separetely from rcnn, so that we must define loss function. We want to maximize iou between predicted regions and bounding boxes on image.
This is many to many relations. And we don't know true labels. As soon as there is a confusion. 
To which bounding box single region proposal belongs to?
What we consider positive region, for which we want maximize probability and improve coordinates? What we consider negative region proposal for which we want minimize probability? Let's note that we care about regression loss only for positive samples!

So positive region proposals are:
- Any region proposal that has iou >= 0.7 with any ground truth bounding box
- The anchor that has maximal uou with ground truth box. For every ground truth box.

Negative region proposals are:
- Any region proposal that has iou <= 0.3 for all ground truth boxes

Non negative and non-positive anchors are ignored.