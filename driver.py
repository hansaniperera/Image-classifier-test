# MODEL
import classifier as classifier
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(300, 300, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())


model.add(Dense(output_dim=128, activation='relu'))
model.add(Dense(output_dim=1, activation='sigmoid'))

# COMPILE
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(300, 300),  # all images will be resized to 300x300
        batch_size=8,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
test_set = test_datagen.flow_from_directory(
        'validation',
        target_size=(300, 300),
        batch_size=2,
        class_mode='binary')

# TRAINING
model.fit_generator(
        training_set,
        steps_per_epoch=50,
        epochs=2,
        validation_data=test_set,
        validation_steps=100)


test_im = image.load_img('test/tilt15.png', target_size=(300, 300))
test_im = image.img_to_array(test_im)
test_im = np.expand_dims(test_im, axis=0)
result = model.predict(test_im)
training_set.class_indices
if result[0][0] >= 0.5:
        prediction = 'drowning'
else:
        prediction = 'not drowning'

print(prediction)

