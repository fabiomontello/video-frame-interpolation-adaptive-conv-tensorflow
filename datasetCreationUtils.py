import tensorflow as tf
import cv2
import os
import random
import numpy as np
import pandas as pd

#function to crop a square image of 150 x 150 around a given pixel
def crop_image(path, cx, cy):
    
  
  #try to read the image (to make sure it exists)
  try:
    img = cv2.imread(path)
  except:
    print('Cannot read: ',path)
    return []

  return img[int(cx-75):int(cx+75), int(cy-75):int(cy+75),:]

# Function which takes every triplet of images in the dataset and extract 25 
# random patches of 150 x 150. It stores them in a folder that will be then compressed
# and saves the files path in a pandas dataframe, which will be used later to load 
# the data in tensorflow
def build_dataset(): 
  data = pd.DataFrame(columns=['frameA', 'frameB','frameC', 'x', 'y'])
  idx = 0
  NUM_OF_CROPS = 25
  for fold in os.listdir('vimeo_interp_test/target/'):
    for elem in os.listdir('vimeo_interp_test/target/'+fold):
      direc = 'vimeo_interp_test/target/'+fold +'/'+elem+'/'
      sh = cv2.imread(direc+'im1.png').shape[0:2]

      x = np.random.randint(75, sh[1] - 75, NUM_OF_CROPS)
      y = np.random.randint(75, sh[0] - 75, NUM_OF_CROPS)

      for j in range(NUM_OF_CROPS):       
        imgA = crop_image(direc+'im1.png', x[j], y[j])
        imgB = crop_image(direc+'im2.png', x[j], y[j])
        imgC = crop_image(direc+'im3.png', x[j], y[j])

        path = './crops_data/'+str(idx)+'/'
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(path +'a.jpg', np.float32(imgA))
        cv2.imwrite(path +'b.jpg', np.float32(imgB))
        cv2.imwrite(path +'c.jpg',  np.float32(imgC))

        data.loc[idx] = ['./crops/'+str(idx)+'/a.jpg','./crops/'+str(idx)+'/b.jpg', './crops/'+str(idx)+'/c.jpg', x[j], y[j]]
        idx +=1
        if(idx%1000 == 0):
          print(idx)
  return data
