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
***

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


#### RPN bounding box regression
***

Regression problem is not scale invariant by definition, becuase images have different shape. But we can turn it into scale invariant problem with simple parametrization trick. 
Instead of predicting absolute values of proposal region or absolute displacement we can use parametrization
```
t_x = (x - x_a) / w_a
t^*_x = (x^* - x_a) / w_a
```
where `x`, `x_a` and `x^*` are for predicted, anchor, and ground truth bounding box.

It follows from equation
```
x = t_x * w_a + x_a
```
so our aim to predict `t_x` that is equivalent to some fraction of width. This `t` parameter is defined with respect to some anchor. As soon as there definite number of anchors it is not hard for model to learn some scale of `t`.

For width and height there's an interesting trick, we use logarithm scale.
```
t_w = log(w/w_a) = log(w) - log(w_a)
```
so that
```
e^{t_w} * w_a = w
```
0 < t_w < 1 will result in w in range [w_a, e * w_a]
-\infty < t_w < 0 will result in w in range [0, 1 * w_a] 
t_w > 1 will result in w in range [w_a, \infty]

But why not just use absolute ratios? 
It is possible to use just ratio w/w_a, but log ratio is good normalization term.
Consider w = 600 and w_a = 20 then t_w = 30, the same parameter t_w is equal to 3.4 in log ratio.

It also prevents from predicting negative width and heights.


### RPN-Params
***

1. Anchors scales. It represents square root of anchor square in image dimension. 
For example if scale=128, then are of anchor rectangle is 16384

2. Aspect ratios (by contract `a = h/w = H/W` and `HW = 1`). It represents ratios of anchor rectangle. Where `h,w` actual sides of anchor, and `H, W` are normalized sizes.
To compute actual size of sides, solve:
```
1. HW = 1 => W = 1/H
2. (Use substitution from (1)) h/w = H^2    => a = H^2
                                            => H = sqrt(a)
                                            => W = 1 / H = 1 / sqrt(a)
```
Now let's note that we have following relation:
```
1. S*H * S*W = S^2
2. S*H / S*W = h/w = H/W
```
because `H*W = 1`, and also `SH` equal to desirable height and `SW` equal to desirable width. 
These height and width preserve given scale and given aspect ratio.


## Open questions
***

Faster RCNN paper computes transformation targets w.r.t. to center of the screen. 
It looks like parameterization described above will also work.
But will it suffer from numerical instability?

- Yes it will work, but may include some biasness, so center coordinates are better.