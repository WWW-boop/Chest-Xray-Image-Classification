# Pneumonia Classification from Chest X-ray Images 🩻🔍
## Contributors 👤
Wikran Petsuwan 6610110277
## Dataset 🖼️
Chest X-Ray Images (Pneumonia)


[Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/)


The  ```normal```  and  ```pneumonia```  folders within the  ```data/image/```  directory contain a substantial number of chest X-ray images for both normal and pneumonia cases.

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

## Training the Model 🏋️‍♂️
To train the model, use the following command:
```bash
history = model.fit(train, epochs=24, validation_data=val, callbacks=[tensorboard_callback])
```

## Model Evaluation 📊
After training, the model's precision, recall, and accuracy can be evaluated using the test dataset.
```bash
print('Precision: ', pre.result().numpy())
print('Recall: ', rec.result().numpy())
print('Accuracy: ', acc.result().numpy())
```
## Testing the Model 🧪
Test the model by using the provided ```predict``` function:
```bash
predict('data/image/NORMAL/IM-0001-0001.jpeg')  # Output: Normal
predict('data/image/PNEUMONIA/person1_bacteria_1.jpeg')  # Output: Pneumonia
```

## Save and Load Model ⬇️
Save the trained model:
```bash
model.save('models/main_model.keras')
```
Load the saved model:
```bash
new_model = load_model('models/main_model.keras')
```
