# AbdomenNet
## Kaggle RSNA 2023 Abdominal Trauma Detection

The AbdomenNet is designed to detect several potential injuries in CT scans of trauma patients from the Kaggle RSNA 2023 Abdominal Trauma Detection dataset. Rapid diagnosis is crucial, as any of these injuries can be fatal in a short time frame if untreated.


### Dataset Description

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


### Python Setup and Usage

```python
!pip install pydicom
!pip install keras
!pip install keras_cv
!pip install keras_core
!pip install tensorflow
!pip install keras_efficientnet

```


### CT Image Processing

- **Loading, Resizing, and Saving DCM Images as PNG:**
  - Load specific slices from a DICOM file using pydicom.
  - Resize them to a size of (256,256)
  - Save them either as original PNG files or normalized PNG files.
- **Normalization Technique:** The normalization was performed using the following formula for each pixel in an image:
  ```python
  normalized_pixel = (original_pixel - np.min(image)) / (np.max(image) - np.min(image))
  ```


### Training

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
  
- **Data Augmentation: RandomCutout("height_factor": 0.2, "width_factor": 0.2), RandomFlip("horizontal"), RandomRotation(0.2)**
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
      # 이미지 증강 파이프라인을 정의
      augmenter = CustomAugmenter(cutout_params={"height_factor": 0.2, "width_factor": 0.2})
  
      # 이미지 증강을 적용
      augmented_images = augmenter(images, training=True)
  
      return (augmented_images, labels)
    ```
  
- **Model Architecture: A pre-trained EfficientNet B3-5 model, with additional layers tailored for this specific task**
  ```python
  def build_model(warmup_steps, decay_steps):
    # Define Input
    inputs = keras.Input(shape=config.IMAGE_SIZE + [3,], batch_size=config.BATCH_SIZE)

    # Define Backbone
    # Use EfficientNetB5 as the backbone
    backbone = keras_efficientnet.EfficientNetB5(include_top=False, input_tensor=inputs, weights='imagenet')
    x = backbone.output

    # GAP to get the activation maps
    gap = keras.layers.GlobalAveragePooling2D()
    x = gap(x)

    # Define 'necks' for each head
    x_bowel = keras.layers.Dense(32, activation='silu')(x)
    x_extra = keras.layers.Dense(32, activation='silu')(x)
    x_liver = keras.layers.Dense(32, activation='silu')(x)
    x_kidney = keras.layers.Dense(32, activation='silu')(x)
    x_spleen = keras.layers.Dense(32, activation='silu')(x)

    # Define heads
    out_bowel = keras.layers.Dense(1, name='bowel', activation='sigmoid')(x_bowel) # use sigmoid to convert predictions to [0-1]
    out_extra = keras.layers.Dense(1, name='extra', activation='sigmoid')(x_extra) # use sigmoid to convert predictions to [0-1]
    out_liver = keras.layers.Dense(3, name='liver', activation='softmax')(x_liver) # use softmax for the liver head
    out_kidney = keras.layers.Dense(3, name='kidney', activation='softmax')(x_kidney) # use softmax for the kidney head
    out_spleen = keras.layers.Dense(3, name='spleen', activation='softmax')(x_spleen) # use softmax for the spleen head

    # Concatenate the outputs
    outputs = [out_bowel, out_extra, out_liver, out_kidney, out_spleen]

    # Create model
    print("[INFO] Building the model...")
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Cosine Decay
    cosine_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=decay_steps,
        alpha=0.0,
        warmup_target=1e-3,
        warmup_steps=warmup_steps,
    )

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=cosine_decay)
    loss = {
        "bowel": keras.losses.BinaryCrossentropy(),
        "extra": keras.losses.BinaryCrossentropy(),
        "liver": keras.losses.CategoricalCrossentropy(),
        "kidney": keras.losses.CategoricalCrossentropy(),
        "spleen": keras.losses.CategoricalCrossentropy(),
    }
    metrics = {
        "bowel": ["accuracy"],
        "extra": ["accuracy"],
        "liver": ["accuracy"],
        "kidney": ["accuracy"],
        "spleen": ["accuracy"],
    }
    print("[INFO] Compiling the model...")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model
  ```


### Inference
