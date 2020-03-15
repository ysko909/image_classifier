# %%
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras
from keras.utils import np_utils
from PIL import Image, ImageOps
from pathlib import Path
import numpy as np

CURRENTDIR = './'

# folders = ['Panzerkampfwagen_iv', 'tiger_i', 'tiger_ii']
folders = ['tiger_i', 'tiger_ii']

# %% [markdown]
# ## 画像の読み込み、水増し、NumPy配列への格納
# `convert('RGB')`してるのは、もともと白黒な画像が含まれているから。


def image_aug(image):
    images = []
    image_size = 100
    image = image.resize((image_size, image_size))
    conv_image = image.convert('RGB')
    images.append(conv_image)
    images.append(ImageOps.flip(conv_image))
    images.append(ImageOps.mirror(conv_image))

    return images


# %% [markdown]
root_folder = Path(CURRENTDIR)

paths = []

X = []
y = []

for i, name in enumerate(folders):
    dir = root_folder / name
    p = Path(str(dir.resolve()))
    for file in p.glob('*.jpeg'):
        images = []
        image = Image.open(file)
        images = image_aug(image)
        for img in images:
            data = np.asarray(img)
            X.append(data)
            y.append(i)

X = np.array(X)
y = np.array(y)

# %% [markdown]
# ## データの正規化
X = X.astype('float32')
X /= 255.0

# %% [markdown]
# ## ラベルをベクトル変換する

y = np_utils.to_categorical(y, len(folders))

# %% [markdown]
# ## 訓練セットとテストセットに分割

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
# ## CNNのモデル構築

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=X_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(folders), activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# %% [markdown]
# ## 学習する

model.fit(X_train, y_train, batch_size=128, epochs=12,
          verbose=1, validation_data=(X_test, y_test))

# %% [markdown]
# ## 評価と結果出力

score = model.evaluate(X_test, y_test, verbose=0)
print('Total loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))


# %%
