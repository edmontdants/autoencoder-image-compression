import imageio
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np


dataset = []
img_indices = range(1, 23)
for index in img_indices:
    img_index = str(index)
    if index < 10:
        img_index = f"0{img_index}"
    filename = f'data/kodim{img_index}.png'
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

x = Conv2D(16, (2, 2), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (2, 2), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print(autoencoder.summary())

# X_train = dataset[:13]
# X_test = dataset[13:]

# Currently the testing is the same as the training set, because the
# dataset is so small so it won't give any results otherwise
autoencoder.fit(dataset, dataset,
                epochs=100,
                batch_size=128,
                shuffle=True)


predicted_imgs = autoencoder.predict(dataset)
for i in range(predicted_imgs.shape[0]):
    plt.imshow(predicted_imgs[i])
    imageio.imsave(f'results/result-{i}.png', predicted_imgs[i])
    plt.show()
