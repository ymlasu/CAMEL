import os
import numpy as np
from PIL import Image

import gzip
import numpy as np
import matplotlib.pyplot as plt


img_list = os.listdir('./coil-100/')
coil_100 = []
for img_name in img_list:
    if img_name == '.DS_Store':
        continue
    I = np.asarray(Image.open(os.path.join('./coil-100/', img_name)))
    coil_100.append(I)
    
coil_100 = np.array(coil_100)
coil_100.shape

plt.imshow(coil_100[5])

coil_100_labels = []
for img_name in img_list:
    if img_name == '.DS_Store':
        continue
    if img_name[4] == '_':
        coil_100_labels.append(int(img_name[3]))
    elif img_name[5] == '_':
        coil_100_labels.append(int(img_name[3:5]))
    else:
        coil_100_labels.append(int(img_name[3:6]))
coil_100_labels = np.array(coil_100_labels)
coil_100_labels.shape


coil_100.dump('coil_100.npy')
coil_100_labels.dump('coil_100_labels.npy')
