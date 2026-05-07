# Pretrained vs From-Scratch CNNs for Robotic Grasp Detection

*Course research paper — Columbia Business School*

> **Status:** stub. Numbers in the Experiments section are placeholders pending the
> Colab training run; replace with values from `results/pretrained/eval.json` and
> `results/scratch/eval.json` once available.

---

## 1. Introduction

Modern warehouse-automation companies (Symbotic, Covariant, Berkshire Grey) and
emerging humanoid manipulation startups (Figure, 1X, Sanctuary) all converge on the
same bottleneck: a robot must look at a novel object and decide where to grip it.
The classical approach — engineered feature pipelines plus exhaustive geometric
search — does not scale to the long tail of SKUs in a real fulfillment center.
Deep learning promises a single network that consumes a camera image and returns
a grasp pose, with the generalization properties that come from training on
diverse data.

Redmon & Angelova [2015] showed that a single AlexNet, regressed end-to-end on
the Cornell Grasping Dataset, can produce a grasp rectangle at 13 fps with
accuracy that beats the prior two-stage baseline. Their result is now standard
methodology, but it predates ImageNet-scale transfer learning becoming the
default. Since then, ResNet-style backbones have eclipsed AlexNet on
classification, and ImageNet-pretrained initialization is the de facto baseline
for downstream vision tasks.

**Research question.** Can a pretrained ResNet-50 effectively predict robotic
grasp rectangles from RGB images, and how does ImageNet-pretrained initialization
compare to training from scratch on the same task?

**Contributions.** We (1) replicate Redmon & Angelova's single-stage CNN
regression formulation on Cornell, swapping AlexNet for ResNet-50; (2) train two
otherwise-identical models — one initialized from ImageNet, one from random
weights — and report the gap on the Cornell metric; (3) discuss what this gap
implies for transfer learning in industrial manipulation systems.

## 2. Related Work

**Two-stage grasp detection.** Lenz, Lee & Saxena [2015] introduced a sliding-window
convolutional pipeline: a small network proposes candidate rectangles and a larger
one ranks them. This was the first deep-learning Cornell result, but the
two-stage architecture costs ~13.5 s per image.

**Single-stage regression.** Redmon & Angelova [2015] reformulated grasp detection
as direct regression: one CNN forward pass produces (x, y, θ, w, h). They report
84.4% accuracy with image-wise split on Cornell at real-time speed (13 fps on a
GPU). Our work follows this formulation.

**Modern grasp networks.** GG-CNN [Morrison et al., 2018] and GR-ConvNet
[Kumra et al., 2020] regress dense per-pixel grasp quality maps from depth or
RGB-D, achieving > 95% on Cornell. We do not include depth; our goal is the
narrower comparison of pretrained vs scratch initialization for the canonical
RGB regression formulation.

## 3. Method

### 3.1 Grasp representation

Following Jiang, Moseson & Saxena [2011], a grasp is an oriented rectangle in the
image, parameterized as $(x, y, \theta, w, h)$ where $(x, y)$ is the centroid,
$\theta$ is the orientation of the gripper-opening axis, $w$ is the gripper
opening, and $h$ is the gripper plate width. The Cornell dataset annotates each
positive grasp as four corner points; we convert each annotation to the 5-tuple
form before training.

### 3.2 Model architecture

We use ResNet-50 [He et al., 2016] as the backbone, replacing the final
1000-class softmax head with a single linear layer mapping the 2048-d global
average pool to 5 outputs. The architecture is identical between the pretrained
and scratch variants; only the initialization differs.

### 3.3 Loss and normalization

We minimize the L2 (mean-squared) error between the predicted and ground-truth
5-tuple. Each dimension is normalized before the loss so all five contribute on
a comparable scale: $x, y$ to $[-1, 1]$ via $2x/W - 1$ and $2y/H - 1$ on the
$224\times224$ input; $\theta$ to $\pm 1$ via $\theta / (\pi/2)$ (gripper
orientation is direction-agnostic, so $\theta$ wraps in $(-\pi/2, \pi/2]$); $w$
and $h$ to $[0, 1]$ via division by the input dimension.

When an image has multiple positive ground-truth grasps, we sample one uniformly
each epoch. Evaluation considers all positive grasps.

### 3.4 Training

Adam, learning rate $10^{-4}$, weight decay $10^{-5}$, batch size 32, up to 30
epochs with early stopping (patience 5) on validation MSE. Augmentation:
random horizontal flip, $\pm 15°$ rotation, color jitter (brightness/contrast/
saturation 0.2, hue 0.05). Random crop was omitted to keep the augmentation
geometry composable with the grasp annotation transform; the rotation
augmentation alone gives the model ~30° of orientation-invariance training
signal beyond the GT annotations.

## 4. Experiments

### 4.1 Dataset and splits

The Cornell Grasping Dataset [Jiang et al., 2011] contains 885 RGB-D images of
240 graspable objects, each annotated with multiple positive (`pcd*cpos.txt`)
and negative (`pcd*cneg.txt`) rectangles. We use only RGB and only the positive
annotations. Native resolution 640×480 is bilinearly resized to 224×224 to
match ResNet input; corner coordinates are scaled by the same factor.

We use a fixed image-wise random 70/15/15 train/val/test split (seeded). This
matches Redmon & Angelova's "image-wise" protocol (as opposed to their
"object-wise" split, which assigns whole object instances to a single fold).
Image-wise is the easier setting; we do not claim object-wise generalization.

### 4.2 Evaluation metric

We report the standard Cornell rectangle metric: a predicted grasp is correct
iff there exists a positive ground-truth grasp for the same image satisfying
both (i) Jaccard index of the oriented rectangles > 0.25 and (ii) absolute
orientation difference < 30°. Primary metric is the percentage of test images
with at least one correct prediction. As a secondary diagnostic, we report the
mean absolute error of each output dimension against the closest ground-truth
grasp by IoU.

### 4.3 Results

> *Pending Colab run — replace with values from `results/{pretrained,scratch}/eval.json`.*

| Model | Init | Cornell acc. | MAE x (px) | MAE y (px) | MAE θ (deg) | MAE w (px) | MAE h (px) |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | ImageNet | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| ResNet-50 | random | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### 4.4 Qualitative results

> See `results/pretrained/qualitative.png` and `results/scratch/qualitative.png`.
> Green: ground-truth positive grasps. Lime/red: model prediction (lime if Cornell-
> correct, red if not).

## 5. Discussion

> *Fill in once numbers are in.*

Anticipated discussion points:
- **Magnitude of the pretrained advantage.** ImageNet-pretrained models typically
  converge faster and to a better optimum on small downstream datasets. Cornell's
  ~620 training images sit squarely in the regime where pretraining matters most.
- **What ImageNet teaches that grasp detection reuses.** Edge, texture, and
  object-boundary features generalize from ImageNet's classification objective
  to Cornell's localization objective even though the loss heads differ.
- **Failure modes.** Inspect qualitative grids for thin/transparent objects,
  cluttered backgrounds, and unusually-shaped grasps that lie outside the
  augmentation envelope.
- **Industry implication.** Real fulfillment-center stacks (Symbotic, Covariant,
  Amazon Robotics) train on millions of in-house images and can reasonably
  bypass ImageNet entirely; for small academic / startup datasets the
  pretrained initialization is essentially free accuracy.

## 6. Conclusion

We replicated Redmon & Angelova's single-stage grasp-regression CNN with a
ResNet-50 backbone and quantified the gap between ImageNet-pretrained and
randomly-initialized variants on the Cornell metric. _Headline number TBD._
Limitations: image-wise split (not object-wise), RGB only (no depth), single
grasp per image (no multi-modal output), ~885-image dataset.

Future directions: (1) object-wise split for stricter generalization; (2)
RGB-D fusion via a separate depth branch; (3) per-pixel grasp-quality output
in the GG-CNN / GR-ConvNet style; (4) sim-to-real transfer with synthetic
grasp data from GraspNet-1Billion or ACRONYM.

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image
  Recognition. *CVPR*.
- Jiang, Y., Moseson, S., & Saxena, A. (2011). Efficient Grasping from RGBD
  Images: Learning Using a New Rectangle Representation. *ICRA*.
- Kumra, S., Joshi, S., & Sahin, F. (2020). Antipodal Robotic Grasping using
  Generative Residual Convolutional Neural Network. *IROS*.
- Lenz, I., Lee, H., & Saxena, A. (2015). Deep Learning for Detecting Robotic
  Grasps. *IJRR* 34 (4-5).
- Morrison, D., Corke, P., & Leitner, J. (2018). Closing the Loop for Robotic
  Grasping: A Real-time, Generative Grasp Synthesis Approach. *RSS*.
- Redmon, J., & Angelova, A. (2015). Real-Time Grasp Detection Using
  Convolutional Neural Networks. *ICRA*. arXiv:1412.3128.
