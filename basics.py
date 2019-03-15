import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation



model=Sequential([
    Dense(32,input_shape=(100,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# Generate dummy data to test on it.
import numpy as np
x_train = np.random.random((1000, 100))
x_test=np.random.random((100,100))
y_test=np.random.randint(10, size=(100,1))
labels = np.random.randint(10, size=(1000, 1))
one_hot_labels1 = keras.utils.to_categorical(labels, num_classes=10)
one_hot_labels2=keras.utils.to_categorical(y_test,num_classes=10)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(x_train, one_hot_labels1, epochs=10, batch_size=32)
score=model.evaluate(x_test,one_hot_labels2,batch_size=128)