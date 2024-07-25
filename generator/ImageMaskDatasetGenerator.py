# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/07/22 ImageMaskDatasetGenerator.py


import os
import cv2
import sys
import glob
import json
import numpy as np
import shutil 
import traceback

class ImageMaskDatasetGenerator:

  def __init__(self, size = 512):
    self.size   = size
    self.RESIZE = (size, size)

  def generate(self, input_dir, output_dir):
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    output_test_dir = os.path.join(output_dir,  "test")
    os.makedirs(output_test_dir)
    
    output_valid_dir = os.path.join(output_dir,  "valid")
    os.makedirs(output_valid_dir)

    output_train_dir = os.path.join(output_dir, "train")
    os.makedirs(output_train_dir)

    valid_image_files = []
    valid_mask_files  = []

    for t in ["A", ]:
      for i in range(100):
        image_file = input_dir + "/test" + t + "_" + str(i+1) + ".bmp"
        if os.path.exists(image_file):
          valid_image_files.append(image_file)
        mask_file = input_dir + "/test" + t + "_" + str(i+1) + "_anno.bmp"
        if os.path.exists(mask_file):
          valid_mask_files.append(mask_file)

    test_image_files = []
    test_mask_files  = []
    for t in ["B"]:
      for i in range(100):
        image_file = input_dir + "/test" + t + "_" + str(i+1) + ".bmp"
        if os.path.exists(image_file):
          test_image_files.append(image_file)
        mask_file = input_dir + "/test" + t + "_" + str(i+1) + "_anno.bmp"
        if os.path.exists(mask_file):
          test_mask_files.append(mask_file)


    train_image_files = []
    train_mask_files  = []
    for i in range(100):
        image_file = input_dir + "/train" + "_" + str(i+1) + ".bmp"
        if os.path.exists(image_file):
          train_image_files.append(image_file)
        mask_file = input_dir + "/train" +  "_" + str(i+1) + "_anno.bmp"
        if os.path.exists(mask_file):
          train_mask_files.append(mask_file)
    
    output_test_images_dir = os.path.join(output_test_dir, "images")
    output_test_masks_dir  = os.path.join(output_test_dir, "masks")
    self.generate_one(test_image_files, test_mask_files,  output_test_images_dir, output_test_masks_dir)

    output_valid_images_dir = os.path.join(output_valid_dir, "images")
    output_valid_masks_dir  = os.path.join(output_valid_dir, "masks")
    self.generate_one(valid_image_files, valid_mask_files, output_valid_images_dir, output_valid_masks_dir)

    output_train_images_dir = os.path.join(output_train_dir, "images")
    output_train_masks_dir  = os.path.join(output_train_dir, "masks")
    self.generate_one(train_image_files, train_mask_files, output_train_images_dir, output_train_masks_dir)

  def generate_one(self, image_files, mask_files,  output_images_dir, output_masks_dir):
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    num_image_files = len(image_files)
    num_mask_files = len(mask_files)
    print("--- image_files len {}".format(num_image_files))
    print("--- mask_files len {}".format(num_mask_files))

    if num_image_files != num_mask_files:
      raise Exception("Unmatched number of the image and mask files")
    
    for i in range(num_image_files):
      image_file = image_files[i]
      image = cv2.imread(image_file)
      # Get the width and height of the original image 
      width =  image.shape[1]
      height = image.shape[0]
      print("--- image width:{} height:{}".format(width, height))
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      basename = os.path.basename(image_file)
      basename = basename.replace(".bmp", ".jpg")
      
      # Resize the image to be self.RESIZE 
      #image = cv2.resize(image, self.RESIZE)
      output_image_filepath = os.path.join(output_images_dir, basename)
      cv2.imwrite(output_image_filepath, image)
      print("--- Saved {}".format(output_image_filepath))

      mask_file = mask_files[i]
      mask = cv2.imread(mask_file)
      mask = mask * 255

      basename = os.path.basename(mask_file)
      basename = basename.replace("_anno.bmp", ".jpg")
      
      # Resize the image to be self.RESIZE 
      output_mask_filepath = os.path.join(output_masks_dir, basename)
      cv2.imwrite(output_mask_filepath, mask)
      print("--- Saved {}".format(output_mask_filepath))
   

if __name__ == "__main__":
  try:
     images_dir = "./Warwick_QU_Dataset"
     output_dir = "../GlaS-MICCAI2015"
     generator = ImageMaskDatasetGenerator()
     generator.generate(images_dir,output_dir)

  except:
    traceback.print_exc()
        
