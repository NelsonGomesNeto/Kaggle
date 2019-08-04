# Imports
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
folder = "D:\ProgrammingBigFiles\Kaggle\Digit Recognizer\\"

# Loading database
traindb = pd.read_csv(folder + "train.csv")
testdb = pd.read_csv(folder + "test.csv")
x_train, y_train = traindb.iloc[:, 1:], traindb.iloc[:, 0]
x_test = testdb
x_train, x_test = x_train / 255.0, x_test / 255.0

sns.countplot(y_train)
plot.show()

input()
tf.keras.layers.Conv2