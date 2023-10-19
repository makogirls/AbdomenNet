# Kaggle RSNA 2023 Abdominal Trauma Detection

This model is designed to detect potential injuries in CT scans of trauma patients using a modified ResNet50 architecture. The model considers both image data from the dicom files and metadata about patients.


## Dataset Description

- `train.csv`: Target labels for the train set. Note that patients labeled healthy may still have other medical issues, such as cancer or broken bones, that don't happen to be covered by the competition labels.
  - `patient_id`: A unique ID code for each patient.
  - `[bowel/extravasation]_[healthy/injury]`: The two injury types with binary targets.
  - `[kidney/liver/spleen]_[healthy/low/high]`: The three injury types with three target levels.
  - `any_injury`: Whether the patient had any injury at all.
- `[train/test]_images/[patient_id]/[series_id]/[image_instance_number].dcm`: The CT scan data, in DICOM format. Scans from dozens of different CT machines have been reprocessed to use the run-length encoded lossless compression format but retain other differences such as the number of bits per pixel, pixel range, and pixel representation. Expect to see roughly 1,100 patients in the test set.
- `[train/test]_series_meta.csv`: Each patient may have been scanned once or twice. Each scan contains a series of images.
  - `patient_id`: A unique ID code for each patient.
  - `series_id`: A unique ID code for each scan.
  - `aortic_hu`: The volume of the aorta in Hounsfield units. This acts as a reliable proxy for when the scan was. For a multiphasic CT scan, the higher value indicates the late arterial phase.
  - `incomplete_organ`: True if one or more organs wasn't fully covered by the scan. This label is only provided for the train set.
- `sample_submission.csv`: A valid sample submission. Only the first few rows are available for download.
- `image_level_labels.csv`: Train only. Identifies specific images that contain either bowel or extravasation injuries.
  - `patient_id`: A unique ID code for each patient.
  - `series_id`: A unique ID code for each scan.
  - `instance_number`: The image number within the scan. The lowest instance number for many series is above zero as the original scans were cropped to the abdomen.
  - `injury_name`: The type of injury visible in the frame.
- `segmentations/`: Model-generated pixel-level annotations of the relevant organs and some major bones for a subset of the scans in the training set. This data is provided in the nifti file format. The filenames are series IDs. Note that the NIFTI files and DICOM files are not in the same orientation. Use the NIFTI header information along with DICOM metadata to determine the appropriate orientation.
- `[train/test]_dicom_tags.parquet`: DICOM tags from every image, extracted with Pydicom. Provided for convenience.


## Python Setup and Usage

```python
!pip install pydicom
!pip install torch
!pip install torch.nn
!pip install torchvision.models
```


## Dataset Processing

- **Loading, Resizing, and Data Augmentation:**
  - Load specific slices from a DICOM file using pydicom.
  - Resize them to a size of (225,225)
  - Data Augmentation: ShiftScaleRotate, Horizontal/Vertical Flip
```python
self.transform = Compose([
              Resize(height=225, width=225),
              ShiftScaleRotate(rotate_limit=90, scale_limit=[0.8, 1.2]),
              HorizontalFlip(p=self.horizontal_flip),
              VerticalFlip(p=self.vertical_flip),
              ToTensorV2()
          ])
```


## Training

- **Define the configuration:**
  ```python
  class Config:
    SEED = 42
    IMAGE_SIZE = [256, 256]
    BATCH_SIZE = 16 or 32 or 64
    EPOCHS = 100 or 200
    TARGET_COLS  = [
        "bowel_injury", "extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ]
    AUTOTUNE = tf.data.AUTOTUNE
  ```
  
- **Data Augmentation: RandomCutout, RandomFlip, RandomRotation**
  ```python
  random_flip_layer = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
  random_rotation_layer = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)

  class CustomAugmenter(tf.keras.layers.Layer):
      def __init__(self, cutout_params, **kwargs):
          super(CustomAugmenter, self).__init__(**kwargs)
          self.cutout_layer = keras_cv.layers.Augmenter([keras_cv.layers.RandomCutout(**cutout_params)])
  
      def call(self, inputs, training=None):
          if training:
              inputs = random_flip_layer(inputs)
              inputs = random_rotation_layer(inputs)
              inputs = self.cutout_layer(inputs)
          return inputs
  
  def apply_augmentation(images, labels):
      augmenter = CustomAugmenter(cutout_params={"height_factor": 0.2, "width_factor": 0.2})
      augmented_images = augmenter(images, training=True)
  
      return (augmented_images, labels)
    ```
  
- **Model Architecture 1: Multi-output Model Architecture - Predicting all 5 labels with a Single EfficientNet B3-5 Model**
  - Uses pretrained EfficientNetB3-5 as the backbone.
  - Contains multiple 'necks' and 'heads' for each label.
  - Employs Global Average Pooling after the backbone.
  - Utilizes Cosine Decay for the learning rate.
  - Different activation functions for binary (sigmoid) and multi-class (softmax) outputs.
  - Uses Binary Crossentropy for binary labels and Categorical Crossentropy for multi-class labels.

- **Model Architecture 2: Single-output Model Architectures - Predicting 1 label per model using EfficientNet B3-5**
  - Binary Classification Model
    - Uses pretrained EfficientNetB3-5 as the backbone.
    - Specifically designed for binary classification tasks.
    - Employs Global Average Pooling after the backbone.
    - Utilizes Cosine Decay for the learning rate.
    - Uses Binary Crossentropy as the loss function.
  - Tertiary Classification Model
    - Uses pretrained EfficientNetB3-5 as the backbone.
    - Specifically designed for 3-class classification tasks.
    - Employs Global Average Pooling after the backbone.
    - Utilizes Cosine Decay for the learning rate.
    - Uses Categorical Crossentropy as the loss function.

- **Train models and save:**
  - Define the model.
  - Train with train dataset and validate with validation dataset for the number of epochs.
  - Save the model.


## Inference
- **Define the configuration for inference:**
  ```python
  class Config:
    IMAGE_SIZE = [256, 256]
    RESIZE_DIM = 256
    BATCH_SIZE = 16 or 32 or 64
    AUTOTUNE = tf.data.AUTOTUNE
    TARGET_COLS  = ["bowel_healthy", "bowel_injury", "extravasation_healthy",
                   "extravasation_injury", "kidney_healthy", "kidney_low",
                   "kidney_high", "liver_healthy", "liver_low", "liver_high",
                   "spleen_healthy", "spleen_low", "spleen_high"]
  ```

- **Load Pre-trained Models and Make Predictions:**
  - Load the model.
  - Predict the condition of the organ for each patient.
  - Store predictions in a dataframe.

