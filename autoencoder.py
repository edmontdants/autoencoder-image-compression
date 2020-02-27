import imageio
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


cwd = Path('//ad.tuni.fi/home/toikkal/StudentDocuments/Documents/media-analysis')
print(cwd)
dataset = []
img_indices = range(1, 23)
for index in img_indices:
    img_index = str(index)
    filename = cwd / 'data' / f'kodim{img_index}.png'
    print(f"Reading {filename}...")
    try:
        img_data = imageio.imread(filename)
    except Exception:
        print("Could not read file, skipping...")
        continue

    print(f"Image shape: {img_data.shape}")
    dataset.append(img_data/255)

dataset = np.array(dataset)
print(dataset.shape)

input_img = Input(shape=dataset[0].shape)

x = Conv2D(16, (7, 7), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(10, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(10, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (7, 7), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (7, 7), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

# X_train = dataset[:13]
# X_test = dataset[13:]

# Currently the testing is the same as the training set, because the
# dataset is so small so it won't give any results otherwise
autoencoder.fit(dataset, dataset,
                epochs=1000,
                batch_size=2,
                shuffle=True)


predicted_imgs = autoencoder.predict(dataset)
for i in range(predicted_imgs.shape[0]):
    plt.figure()
    img_uint = (255*predicted_imgs[i]).astype(np.uint8)
    plt.imshow(predicted_imgs[i])
    imageio.imsave(cwd / 'results' / f'result-{i+1}.png', img_uint)
plt.show()
