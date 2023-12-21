import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import random
import re

from tqdm import tqdm

import pydicom as dicom
import nibabel as nib

import pydicom
from pydicom.data import get_testdata_files
import plotly.express as px
from pprint import pprint
import nibabel as nib
from ipywidgets import interact
DS_RATE = 2

class Medical3DImageVisualizer:
    def __init__(self, downsample_rate=1):
        """ 
        초기화 함수.
        
        Parameters:
        - downsample_rate: 이미지의 다운샘플링 비율. 기본값은 1 (다운샘플링 없음).
        """
        self.downsample_rate = downsample_rate

    def create_3D_scans(self, folder):
        """ 
        주어진 폴더에서 DICOM 이미지들을 읽어 3D 볼륨으로 변환합니다.
        
        Parameters:
        - folder: DICOM 이미지가 저장된 폴더 경로.

        Returns:
        - volume: 3D 볼륨 데이터.
        """
        filenames = os.listdir(folder)
        filenames = [int(filename.split('.')[0]) for filename in filenames]
        filenames = sorted(filenames)
        filenames = [str(filename) + '.dcm' for filename in filenames]
        
        volume = []
        for filename in tqdm(filenames[::self.downsample_rate]):
            filepath = os.path.join(folder, filename)
            ds = dicom.dcmread(filepath)
            image = ds.pixel_array
            
            # Rescale 파라미터 찾기
            if ("RescaleIntercept" in ds) and ("RescaleSlope" in ds):
                intercept = float(ds.RescaleIntercept)
                slope = float(ds.RescaleSlope)
        
            # Clipping 파라미터 찾기
            center = int(ds.WindowCenter)
            width = int(ds.WindowWidth)
            low = center - width / 2
            high = center + width / 2    
            
            image = (image * slope) + intercept
            image = np.clip(image, low, high)
            image = (image / np.max(image) * 255).astype(np.int16)
            image = image[::self.downsample_rate, ::self.downsample_rate]
            volume.append(image)
        
        volume = np.stack(volume, axis=0)
        return volume

    def create_3D_segmentations(self, filepath):
        """ 
        NIfTI 파일에서 분할 데이터를 읽어 3D 볼륨으로 변환합니다.
        
        Parameters:
        - filepath: NIfTI 분할 데이터 파일 경로.

        Returns:
        - img: 3D 분할 볼륨 데이터.
        """

        img = nib.load(filepath).get_fdata()
        img = np.transpose(img, [1, 0, 2])
        img = np.rot90(img, 1, (1,2))
        img = img[::-1, :, :]
        img = np.transpose(img, [1, 0, 2])
        img = img[::self.downsample_rate, ::self.downsample_rate, ::self.downsample_rate]
        return img

    def plot_image_with_seg(self, volume, volume_seg=[], orientation='Coronal', num_subplots=20):
        """ 
        주어진 3D 볼륨에서 2D 슬라이스를 선택하고, 이미지와 분할 영역을 함께 시각화합니다.
        
        Parameters:
        - volume: 3D 볼륨 데이터.
        - volume_seg: 3D 분할 볼륨 데이터. 기본값은 빈 리스트.
        - orientation: 시각화할 슬라이스의 방향 ('Coronal', 'Sagittal', 'Axial'). 기본값은 'Coronal'.
        - num_subplots: 시각화할 슬라이스 수. 기본값은 20.
        """

        if len(volume_seg) == 0:
            plot_mask = 0
        else:
            plot_mask = 1

        # 원하는 방향으로 슬라이스 선택
        if orientation == 'Coronal':
            slices = np.linspace(0, volume.shape[2]-1, num_subplots).astype(np.int16)
            volume = volume.transpose([1, 0, 2])
            if plot_mask:
                volume_seg = volume_seg.transpose([1, 0, 2])
        elif orientation == 'Sagittal':
            slices = np.linspace(0, volume.shape[2]-1, num_subplots).astype(np.int16)
            volume = volume.transpose([2, 0, 1])
            if plot_mask:
                volume_seg = volume_seg.transpose([2, 0, 1])
        elif orientation == 'Axial':
            slices = np.linspace(0, volume.shape[0]-1, num_subplots).astype(np.int16)

        # 그림을 그리기 위한 설정
        rows = np.max([np.floor(np.sqrt(num_subplots)).astype(int) - 2, 1])
        cols = np.ceil(num_subplots/rows).astype(int)
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 4))
        fig.tight_layout(h_pad=0.01, w_pad=0)
        ax = ax.ravel()
        for this_ax in ax:
            this_ax.axis('off')
        
        # 슬라이스를 순서대로 시각화
        for counter, this_slice in enumerate(slices):
            plt.sca(ax[counter])
            image = volume[this_slice, :, :]
            plt.imshow(image, cmap='gray')
            if plot_mask:
                mask = np.where(volume_seg[this_slice, :, :], volume_seg[this_slice, :, :], np.nan)
                plt.imshow(mask, cmap='Set1', alpha=0.5)       

    def visualize(self, dcm_folder, nii_filepath, orientation='Coronal', num_subplots=20):
        """ 
        DICOM 폴더와 NIfTI 파일을 읽어들여 시각화를 수행합니다.
        
        Parameters:
        - dcm_folder: DICOM 이미지가 저장된 폴더 경로.
        - nii_filepath: NIfTI 분할 데이터 파일 경로.
        - orientation: 시각화할 슬라이스의 방향 ('Coronal', 'Sagittal', 'Axial'). 기본값은 'Coronal'.
        - num_subplots: 시각화할 슬라이스 수. 기본값은 20.
        """
        volume = self.create_3D_scans(dcm_folder)
        volume_seg = self.create_3D_segmentations(nii_filepath)
        
        print(f'3D Image file shape: {volume.shape}')
        print(f'3D segmentation file shape: {volume_seg.shape}')
        
        self.plot_image_with_seg(volume, volume_seg, orientation, num_subplots)

    def visualize_all_orientations(self, dcm_folder, nii_filepath, num_subplots=20):
        """ 
        DICOM 폴더와 NIfTI 파일을 읽어들여 모든 방향에 대한 시각화를 수행합니다.
        
        Parameters:
        - dcm_folder: DICOM 이미지가 저장된 폴더 경로.
        - nii_filepath: NIfTI 분할 데이터 파일 경로.
        - num_subplots: 시각화할 슬라이스 수. 기본값은 20.
        """
        volume = self.create_3D_scans(dcm_folder)
        volume_seg = self.create_3D_segmentations(nii_filepath)

        print(f'3D Image file shape: {volume.shape}')
        print(f'3D segmentation file shape: {volume_seg.shape}')
        
        orientations = ['Coronal', 'Sagittal', 'Axial']
        
        for orientation in orientations:
            print(f"Showing {orientation} orientation")
            self.plot_image_with_seg(volume, volume_seg, orientation, num_subplots)

if __name__ == "__main__":
    visualizer = Medical3DImageVisualizer(downsample_rate=1)
    visualizer.visualize_all_orientations('train_images/10004/21057', 'segmentations/21057.nii', 10)
