# Judo Belt Detection and Grayscale Conversion — Case Study

> [!NOTE]
> This repository contains technical documentation for a computer-vision
> project. It describes the design, development process, and qualitative
> observations from the project.
>
> The repository does not distribute the source photographs, annotations,
> dataset, trained model weights, generated outputs, or implementation files.
> It is intended as a case study rather than a runnable software package.

## Abstract

This case study describes a computer-vision pipeline developed to detect belts
worn by judo athletes and selectively convert the detected belt regions to
grayscale.

The pipeline uses a custom-trained Ultralytics YOLO instance-segmentation model
to produce pixel-level masks of visible belts. OpenCV and NumPy are then used
to modify only the pixels inside the predicted masks while preserving local
brightness variation, shadows, fabric folds, and texture.

A separate post-processing step handles black belts. Because black pixels are
already achromatic, ordinary grayscale conversion causes little or no visible
change. Dark, low-saturation pixels inside the predicted belt mask are
therefore adjusted toward a visible medium-gray intensity while retaining some
of their original brightness variation.

This document covers the project's motivation, annotation methodology, dataset
construction, model training, mask-based image processing, qualitative
observations, and known limitations.

---

## 1. Project Goal

Given a photograph containing one or more judo athletes, the pipeline was
designed to:

1. Detect each visible judo belt.
2. Produce a pixel-level segmentation mask for each detected belt.
3. Convert the pixels inside each belt mask to grayscale.
4. Leave the remainder of the photograph unchanged.
5. Preserve the belt's folds, shadows, and fabric texture.
6. Save the edited photograph to a separate output directory.

The intended approach needed to handle multiple belt colors, including
visually challenging cases such as white and black belts.

---

## 2. Why Instance Segmentation Was Used

Two simpler approaches were considered before using instance segmentation.

### 2.1 Color-Based Thresholding

A color-based method could search for blue, brown, black, white, or other belt
colors. However, similar colors can also appear in:

- The judogi
- Competition mats
- Shadows
- Background objects
- Other clothing

White and black belts are particularly difficult to isolate using color alone
because they may blend into uniforms, shadows, and background regions.

### 2.2 Generic Pretrained Object Detection

General-purpose pretrained detection models can identify classes such as
`person`, but commonly used datasets such as COCO do not include a dedicated
judo-belt class.

A custom model was therefore required to learn the visual appearance of belts
from their shape, texture, position, and surrounding context.

### 2.3 Instance Segmentation

Instance segmentation was selected because it predicts a pixel-level object
mask rather than only a rectangular bounding box.

A bounding box around a belt would also contain parts of the athlete's jacket,
hands, and waist. Using the predicted segmentation mask allows the
image-processing stage to modify a more targeted region.

---

## 3. Pipeline Overview

```text
Judo photographs
        │
        ▼
Manual polygon annotation
        │
        ▼
YOLO segmentation dataset
        │
        ▼
Training / validation split
        │
        ▼
Custom YOLO model training
        │
        ▼
Selected model checkpoint
        │
        ▼
Belt detection and mask inference
        │
        ▼
Mask-constrained OpenCV processing
        │
        ▼
Edited photographs
```

Training, inference, mask creation, and image recoloring were run locally
through Python.

---

## 4. Technology Stack

|
 Component 
|
 Purpose 
|
|
---
|
---
|
|
 Python 
|
 Primary implementation language 
|
|
 Ultralytics YOLO 
|
 Instance-segmentation training and inference 
|
|
 PyTorch 
|
 Underlying machine-learning framework 
|
|
 OpenCV 
|
 Image loading, mask creation, and pixel recoloring 
|
|
 NumPy 
|
 Numerical image and mask operations 
|
|
 CVAT 
|
 Manual polygon annotation and dataset export 
|

The exact package versions used during development are not specified in this
case study.

---

## 5. Initial Model Test

A pretrained YOLO segmentation model was first tested on a judo photograph.

The pretrained model recognized the athlete as a person but did not identify
the belt as a separate object. This was expected because a belt-specific class
was not available in the model's general-purpose pretrained dataset.

This test confirmed that a custom belt class and a domain-specific annotated
dataset were required.

---

## 6. Annotation Methodology

A single object class named `belt` was created in the annotation tool.

Polygon annotations were manually drawn around visible belt regions according
to the following guidelines:

- Annotate every clearly visible belt.
- Follow the physical belt boundaries as closely as possible.
- Include the knot and hanging belt ends.
- Exclude the surrounding judogi, hands, and waist.
- Use separate polygons for disconnected visible belt sections.
- Annotate each athlete's belt independently in multi-person photographs.
- Leave photographs without a visible belt unannotated so they can serve as
  negative examples.

The annotations were exported in Ultralytics YOLO segmentation format.

Each photograph was associated with a text label file using the same base
filename:

```text
example.jpg
example.txt
```

The corresponding text file contained a class identifier followed by normalized
polygon coordinates.

---

## 7. Dataset Construction

### 7.1 Dataset Composition

The dataset was assembled incrementally:

- An initial batch of 10 annotated photographs
- A second batch of 100 annotated photographs
- A combined total of 110 image-label pairs

### 7.2 Training and Validation Split

An approximate 80/20 split was used:

|
 Subset 
|
 Image count 
|
|
---
|
---:
|
|
 Training 
|
 88 
|
|
 Validation 
|
 22 
|
|
**
Total
**
|
**
110
**
|

The training subset was used for model parameter updates.

The validation subset was excluded from parameter updates and used to monitor
performance and select a model checkpoint. No independent test set was used,
so the observations in this case study should not be interpreted as a final
benchmark of model generalization.

### 7.3 Dataset Structure

```text
judo_belt_dataset_v2/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 7.4 Dataset Configuration

The YOLO dataset configuration followed this structure:

```yaml
path: <dataset_root_path>
train: images/train
val: images/val

names:
  0: belt
```

### 7.5 Dataset Preparation

The dataset-preparation stage performed the following operations:

1. Read the initial image-label pairs.
2. Read the additional image-label pairs.
3. Match each photograph with its annotation using the filename.
4. Check for missing and duplicate filenames.
5. Combine all valid image-label pairs.
6. Randomly shuffle the combined dataset.
7. Create the training and validation split.
8. Copy the files into the required YOLO directory structure.
9. Create the `data.yaml` configuration file.

---

## 8. Model Training

A pretrained YOLO segmentation model was fine-tuned using the custom `belt`
class.

The training configuration was similar to:

```python
model.train(
    data="judo_belt_dataset_v2/data.yaml",
    epochs=100,
    imgsz=640,
    batch=2,
    workers=0,
    device="cpu",
)
```

Training was performed locally using CPU hardware.

The exact pretrained YOLO checkpoint used to initialize the final training run
is not specified here.

### 8.1 Training Diagnostics

The diagnostics monitored during training included:

- `box_loss` — bounding-box regression loss
- `seg_loss` — segmentation-mask loss
- `cls_loss` — classification loss
- `dfl_loss` — distribution focal loss

The segmentation loss was particularly relevant because it reflected errors in
the predicted masks.

However, training loss is not a direct measurement of real-world mask quality.
Metrics such as mask precision, recall, intersection over union, and mAP would
be required for a more complete quantitative evaluation.

### 8.2 Model Checkpoints

The training process produced two standard checkpoint files:

- `best.pt` — the checkpoint selected by Ultralytics according to its
  validation fitness calculation and used for downstream inference.
- `last.pt` — the checkpoint from the final training epoch, which can be useful
  when resuming training.

The selected checkpoint was stored in the project as:

```text
models/best_v2.pt
```

Model weights are not distributed in this repository.

---

## 9. Belt Detection and Mask Inference

For each input photograph, the trained model produced:

- A confidence score for each detected belt
- A bounding box for each detection
- A pixel-level segmentation mask

Only the segmentation masks were used to determine which pixels should be
modified. Bounding boxes were treated as auxiliary detection output and were
not used for recoloring.

The inference configuration was similar to:

```python
results = model.predict(
    source=str(input_path),
    conf=0.25,
    imgsz=640,
    retina_masks=True,
    verbose=False,
)
```

The `retina_masks=True` option requests masks aligned with the original image
resolution.

The predicted segmentation polygons were rasterized and combined into a binary
belt mask.

A small amount of mask erosion was also used during development:

```python
SHRINK_PIXELS = 1
```

This slightly tightened the predicted mask to reduce the chance of recoloring
nearby clothing.

---

## 10. Grayscale Conversion

For each photograph, the image-processing pipeline performed the following
steps:

1. Load the photograph using OpenCV.
2. Run the trained segmentation model.
3. Extract the predicted belt polygons.
4. Rasterize the polygons into a binary mask.
5. Combine all detected belt regions into one mask.
6. Optionally erode the mask slightly.
7. Create a grayscale version of the photograph.
8. Copy grayscale pixels into the regions selected by the belt mask.
9. Apply a separate brightness adjustment to probable black-belt pixels.
10. Save the edited photograph to an output directory.

The standard grayscale conversion preserved the original brightness variation
within colored belts. This allowed folds, shadows, and fabric texture to remain
visible rather than replacing the belt with a flat gray color.

---

## 11. Black-Belt Adjustment

Ordinary grayscale conversion does not visibly alter a black belt because
black pixels are already achromatic.

To make black belts visibly gray, pixels were treated as probable black-belt
pixels when they met all of the following conditions:

- The pixel was inside a predicted belt mask.
- Its grayscale brightness was below a configured threshold.
- Its color saturation was below a configured threshold.

The parameter values used during development were:

```python
BLACK_MAX_BRIGHTNESS = 105
BLACK_MAX_SATURATION = 120
BLACK_TARGET_GRAY = 145
BLACK_TEXTURE_STRENGTH = 0.35
```

The average brightness of the selected dark pixels was calculated. Each
selected pixel was then adjusted toward the target gray value while preserving
part of its original deviation from that average.

Conceptually, the adjustment followed this relationship:

```text
adjusted brightness =
    target gray
    + (original brightness - average black-belt brightness)
      × texture strength
```

The adjusted values were clipped to a limited intensity range.

This approach made dark belt regions more visibly gray while retaining some
local brightness variation from folds, highlights, and shadows.

---

## 12. Qualitative Observations

The project was evaluated informally using the available training and
validation photographs. No independent test set was used, and quantitative
performance results are not presented here.

### 12.1 Positive Observations

During informal testing:

- The model frequently detected blue belts in the tested photographs.
- Black belts could be detected and visibly adjusted using the dedicated
  brightness-remapping step.
- The model handled some standing and partially obscured postures.
- Detections were observed against both white and blue judogi.
- The processing pipeline handled directories containing multiple images.
- Grayscale conversion retained visible belt folds, shadows, and texture.

### 12.2 Observed Limitations

The observed limitations included:

- Predicted masks sometimes failed to cover the full physical belt edge.
- Predicted masks occasionally extended into nearby judogi fabric.
- Small, heavily occluded, or motion-blurred belts could be missed.
- Black-belt recoloring required additional processing beyond ordinary
  grayscale conversion.
- Boundary precision depended strongly on annotation consistency and quality.
- The relatively small dataset limited the range of conditions represented.
- No independent quantitative test evaluation was performed.

These observations should be treated as qualitative development findings rather
than independently verified performance measurements.

---

## 13. Mask-Precision Trade-Off

An important trade-off was identified between expanding and contracting the
predicted masks.

### 13.1 Mask Expansion

Expanding a predicted mask can include missing belt-edge pixels, but it can
also include nearby jacket or waist pixels.

### 13.2 Mask Contraction

Contracting a predicted mask can reduce the inclusion of nearby clothing, but
it may leave a thin strip of the belt's original color around the boundary.

This trade-off cannot be completely resolved through post-processing alone.

A more effective long-term approach would be to add carefully annotated
examples representing the observed boundary failures and retrain the model.
The updated model could then be evaluated using mask-quality metrics on a
separate test set.

---

## 14. Data and Privacy

No source photographs, annotations, dataset files, trained model weights, or
generated outputs are included in this repository.

The photographs used for development depicted identifiable individuals and are
not distributed publicly.

Any future implementation involving identifiable individuals should use:

- Appropriately licensed or authorized photographs
- Documented permission for the intended processing
- An approved storage and retention process
- An approved annotation workflow
- Appropriate access controls
- A review process before publishing data, model artifacts, or examples

Sensitive photographs and derived artifacts should not be committed to a
public Git repository without explicit authorization.

---

## 15. Future Work

Possible extensions of the project include:

- Expanding the dataset with more varied photographic conditions
- Including additional belt colors, poses, and lighting conditions
- Adding difficult boundary and occlusion examples
- Including more belt-absent negative examples
- Creating a separate test set for final evaluation
- Reporting mask precision, recall, intersection over union, and mAP
- Comparing different YOLO segmentation model sizes
- Using GPU acceleration for training
- Adding confidence logging
- Routing uncertain detections to a manual-review directory
- Creating before-and-after comparison outputs
- Supporting video input
- Adding automated image-label integrity tests
- Developing a graphical interface for non-technical users
- Recording exact package versions for reproducibility

---

## 16. Repository Status

This repository contains technical documentation only.

It does not include:

- Source photographs
- Annotation files
- Dataset directories
- Trained model weights
- Generated outputs
- Source-code files
- A dependency specification

The material is published as a description of the project's methodology and
development process. It should not be treated as:

- A runnable application
- A reproducible experiment
- An independently validated benchmark
- Production-ready software

---

## 17. Conclusion

This project applied custom instance segmentation to a domain-specific object
that is not represented as a standard class in common general-purpose object
detection datasets.

The pipeline combined:

1. Polygon-based belt annotation
2. Fine-tuning of a YOLO segmentation model
3. Pixel-level mask inference
4. Mask-constrained OpenCV grayscale conversion
5. Dedicated brightness remapping for black belts

During informal testing, the approach produced useful results across several
of the tested belt colors, including black belts. The most significant
limitation was mask-boundary precision, particularly where belts touched
similarly colored clothing or were partially obscured.

Further development would require a larger and more varied authorized dataset,
a separate test set, quantitative mask evaluation, and systematic testing
across different photographic conditions.

---

## Development Note

This project was completed as a personal learning exercise. Generative AI tools
were used to assist with technical planning, code organization,
troubleshooting, and documentation drafting.

The final documentation was reviewed and adapted by the author. The complete
development conversation is not included in this repository.
