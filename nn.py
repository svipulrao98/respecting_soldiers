#import libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import csv

#import dataset
dataset=pd.read_csv('train.csv')

#transform to usable form
X = dataset.iloc[:, 3:23].values
y = dataset.iloc[:, -1].values

#normalize
sc = StandardScaler()
X = sc.fit_transform(X)

#neural  network
#first layer
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 30, init = 'Orthogonal', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 50, init = 'he_uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 80, init = 'lecun_uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 60, init = 'glorot_normal', activation = 'relu'))
classifier.add(Dense(output_dim = 40, init = 'he_normal', activation = 'relu'))
classifier.add(Dense(output_dim = 26, init = 'glorot_uniform', activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, init = 'lecun_normal', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 1000, nb_epoch = 100, validation_split=0.02)

#for testing
dataset=pd.read_csv('test.csv')

#usable form
X_test = dataset.iloc[:, 3:23].values
X_test = sc.transform(X_test)

#predict
y_pred=classifier.predict(X_test)

#save to csv file
predictions=[]
for i in range(len(dataset)):
    predictions.append([dataset['soldierId'][i], y_pred[i][0]])

with open('predictions4.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(predictions)
csvFile.close()




#THANK YOU!

