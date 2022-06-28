import numpy as np

def get_patches(img):
  #patches = extract_patches_2d(img, (patch_size, patch_size))
  return np.array([img[30:110, :80], img[30:110, 85:165], img[30:110, 170:250]])


def ENL(img):  
  patches = get_patches(img) 

  enl = 0
  n = patches.shape[0]

  for i in range(n):
    enl += (np.mean(patches[i])**2 / np.var(patches[i]))

  return enl/n

def EPI_patch(clean, denoised):
  h, w = denoised.shape
  num, den = 0, 0

  for i in range(h-1):
    for j in range(w):
      num += abs(denoised[i+1][j]-denoised[i][j])
      den += abs(clean[i+1][j]-clean[i][j])
  
  return num/den

def EPI(clean, denoised):
  clean_patches = get_patches(clean) 
  denoised_patches = get_patches(denoised) 
  
  assert denoised_patches.shape[0] == clean_patches.shape[0]
  n = denoised_patches.shape[0]

  epi = 0
  for i in range(n):
    epi += EPI_patch(clean_patches[i], denoised_patches[i])
  
  return epi/n