# %%

from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(0)

DIR = './'

# %%
p = Path(DIR)

paths = []
for file in p.glob('**/*.jpeg'):
    paths.append(file)

print('image number: {}'.format(len(paths)))

# %%
paths = paths[:300]
print(len(paths))

# %%


def image_to_matrix(img):
    img_array = np.asarray(img)

    return img_array


def flatten_image(img_array):
    s = img_array.shape[0] * img_array.shape[1]
    img_width = img_array.reshape(1, s)

    return img_width[0]


# %%
dataset = []
for path in paths:
    img = Image.open(str(path.resolve()))

    img = image_to_matrix(img)

    # img = flatten_image(img)

    dataset.append(img)

dataset = np.array(dataset)
print('dataset shape: {}'.format(dataset.shape))

# %%
n = dataset.shape[0]
pca = IncrementalPCA(n_components=100)

for i in range(n):
    r_dataset = pca.partial_fit(dataset[i:(i + 1)])

r_dataset = pca.transform(dataset)
print('r_dataset.shape: {}'.format(r_dataset.shape))
