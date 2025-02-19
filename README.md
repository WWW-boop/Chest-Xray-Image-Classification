# Pneumonia Classification from Chest X-ray Images 🩻🔍
## Contributors 👤
Wikran Petsuwan 6610110277
## Model Architecture 🧠
```bash
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```
This CNN architecture consists of three convolutional layers with max-pooling, followed by a flattening layer and two dense layers. The final layer uses the sigmoid activation function for binary classification.
