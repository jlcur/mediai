from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Creates a machine learning model for predicting heart disease using a public dataset.
Uses Keras, Tensorflow, and scikit-learn.
"""

# Load dataset
dataset = pd.read_csv("datasets/heart.csv")

# Data exploration
# Check if any attributes have missing values
print(dataset.isnull().sum())
print(dataset.isna().sum())
print(dataset.head())

# Split dataset into input and output
input = dataset.drop("target", axis=1)
output = dataset["target"]

# Split dataset into training and testing sets using 30% of data
input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.30, random_state=0)

# Use feature scaling to get attributes within a similar range
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test = scaler.transform(input_test)

# Create a sequential model with 3 layers
model = Sequential()
model.add(Dense(12, input_dim=13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using the training models
history = model.fit(input_train, output_train, validation_split=0.33, epochs=150, batch_size=10)

# Evaluate the model, print the accuracy score
score = model.evaluate(input_test, output_test, verbose=0)
score = score[1] * 100
print("Accuracy of model: %.0f%%" % score)

# Visualize the accuracy of the model
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training Set", "Testing Set"], loc="upper left")
plt.show()

# Save the model for use in the API
model.save("saved/heart_disease_model.h5")



