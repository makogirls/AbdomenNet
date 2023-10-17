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


