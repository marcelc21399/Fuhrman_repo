#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:06:16 2019

@author: marcello
"""

import os
import argparse
import nrrd
import pandas
import numpy as np
import SimpleITK as sitk


def n4_bias_correction(image):
    import ants
    as_ants = ants.from_numpy(image)
    corrected = ants.n4_bias_field_correction(as_ants)
    return corrected.numpy()

def registration(reference, image, segmentation):
    import ants
    reference_as_ants = ants.from_numpy(reference)
    image_as_ants = ants.from_numpy(image)
    #Rigid, Affine, Similarity, SyN
    output = ants.registration(reference_as_ants, image_as_ants, type_of_transform='SyN')
    registered_image = output.get("warpedmovout")
    segmentation_as_ants = ants.from_numpy(segmentation)
    registered_segmentation = ants.apply_transforms(reference_as_ants, segmentation_as_ants, output.get("fwdtransforms"))
    registered_segmentation = registered_segmentation.numpy()
    registered_segmentation[registered_segmentation > 0] = 1
    return registered_image.numpy(), registered_segmentation

def intensity_normalization(reference, image):
    reference_as_sitk = sitk.GetImageFromArray(reference)
    image_as_sitk = sitk.GetImageFromArray(image)
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(128)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    output = matcher.Execute(image_as_sitk, reference_as_sitk)
    return np.array(sitk.GetArrayViewFromImage(output))

def preprocess_pack(reference, images_w_segmentations, use_n4_bias=False, use_registration=False):
    final_images = list()
    if use_n4_bias:
        reference = n4_bias_correction(reference)
    for (image, segmentation) in images_w_segmentations:
        if use_n4_bias:
            image = n4_bias_correction(image)
        if use_registration:
            image, segmentation = registration(reference, image, segmentation)
        image = intensity_normalization(reference, image)
        final_images.append((image, segmentation))
    return reference, final_images