from model2 import *
from data import *
import numpy as np 

images = train('data/membrane/train/image', num_image = 27, as_gray = True)
labels = label('data/membrane/train/label', num_image = 27, as_gray = True)
print(images.shape)
print(labels.shape)
model = unet2()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose = 1, save_best_only = True)
model.fit(images, labels, validation_split =0.03, batch_size=128, epochs = 10, callbacks=[model_checkpoint])

tests = test('data/membrane/test', num_image = 3, as_gray = True)
results = model.predict(tests, verbose=1)

saveResult("data/membrane/test",results)