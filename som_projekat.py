import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

np.set_printoptions(precision = 4, suppress=True)

data = pd.read_csv(r"C:\Users\aleks\OneDrive\Documents\ETF\03 - treca godina\sesti semestar\SOM\projekat\20_covid_dataset.csv")

data.info(verbose = True)
#c data.head()

#kodovanje 'Yes' u 1 i 'No' u 0
data = data.replace(to_replace = 'Yes',value = 1)
data = data.replace(to_replace = 'No',value = 0)

data.info(verbose = True) 
#c data.describe()
#c data.isnull().any() - provera da li je ostala neka null vrednost

data.hist(figsize=(20,15))
plt.show()

#vidimo da su dva atributa skroz neinformativna (Wearing Masks)
#i (Sanitization from Market) posto imaju uvek istu vrednost, pa cemo ih odmah
#izbaciti

covid = data.copy(deep = True)

covid.pop('Wearing Masks')
covid.pop('Sanitization from Market') 

#ostala neinformativna obelezja cemo izbaciti tako sto cemo posmatrati
#korelaciju (Pearson-ov metod) svakog obelezja sa poslednjom kolonom iz dataseta
#koja nam oznacava da li je osoba pozitivna ili negativna i zadrzacemo onih 10 
#obelezja sa najvecim koeficijentom korelacije

#%% korelacija svih obelezja
pearson_R = covid.corr(method='pearson') #neophodno da prvo odredimo korelaciju
#preostalih obelezja, zatim da te vrednosti sortiramo, a onda da 10 obelezja sa 
#najvecom korelacijom sacuvamo, a ostale odbacimo

correlation = pd.Series(pearson_R.iloc[:,-1])
correlation = correlation.sort_values(ascending=True)
print(correlation)

#izbacivanje obelezja sa najmanjom korelacijom

for i in range(0,8):
    covid.pop(correlation.index[i])

covid_copy = covid.copy(deep = True)
covid_copy.pop('COVID-19')

pearson_R = covid_copy.corr(method='pearson')
plt.figure()
sb.heatmap(pearson_R, annot=True) #prikaz 10 obelezja
plt.show()

spearman_R = covid_copy.corr(method='spearman')
plt.figure()
sb.heatmap(spearman_R, annot=True) #prikaz 10 obelezja
plt.show()

#%% Information Gain

def calculateInfoD(col):
    un = np.unique(col)
    infoD = 0
    for u in un:
        p = sum(col == u)/len(col)
        infoD -= p*np.log2(p)
    return infoD

klasa = covid.iloc[:,-1]

infoD = calculateInfoD(klasa)
print('Info(D) = ' + str(infoD))

feature_names = covid.columns

IG = np.zeros((covid.shape[1]-1, 2))
for ob in range(covid.shape[1]-1):
    f = np.unique(covid.iloc[:, ob])
    
    infoDA = 0
    for i in f:
        temp = klasa[covid.iloc[:, ob] == i]
        
        infoDi = calculateInfoD(temp)
        Di = sum(covid.iloc[:, ob] == i)
        D = len(covid.iloc[:, ob])
        
        infoDA += Di*infoDi/D
    
    IG[ob, 0] = ob+1
    IG[ob, 1] = infoD - infoDA
    
    print(str(feature_names[ob]) + ' IG = ' + str(IG[ob,1]))
    print('------')
    
print('IG = \n' + str(IG))
IGsorted = IG[IG[:, 1].argsort()]
print('Sortirano IG = \n' + str(IGsorted))

#%% LDA

sb.set(style = "darkgrid")

X_pom = covid.iloc[:,:-1]
X = X_pom - np.mean(X_pom,axis = 0)
X /= np.max(X, axis = 0)

y = covid.iloc[:,-1]

X0 = X.loc[y==0,:]
X1 = X.loc[y==1,:]
M0 = X0.mean().values.reshape(X0.shape[1],1)
M1 = X1.mean().values.reshape(X1.shape[1],1)
Sx0 = X0.cov()
Sx1 = X1.cov()
p0 = X0.shape[0]/X.shape[0]
p1 = X1.shape[0]/X.shape[0]

Sw = p0*Sx0 + p1*Sx1
M = p0*M0 + p1*M1
Sb = p0*(M0-M)@(M0-M).T + p1*(M1-M)@(M1-M).T
Sm = Sw + Sb

S1 = Sw
S2 = Sb 

S = np.linalg.inv(S1)@S2
[eigval,eigvec] = np.linalg.eig(S)

ind = np.argsort(eigval)[::-1]
eigval_sort = eigval[ind]
eigvec_sort = eigvec[:,ind]

#%% LDA - 2 dimenzije

A = eigvec_sort[:,:2]
Y = A.T@X.T

data_LDA_2 = pd.DataFrame(data = Y.T)
data_LDA_2 = pd.concat([data_LDA_2,y],axis = 1)

data_LDA_2.columns = ['LDA1', 'LDA2', 'COVID19']

plt.figure()
sb.scatterplot(data = data_LDA_2, x = 'LDA1', y = 'LDA2' , hue = 'COVID19')
plt.show()

#%% LDA - 3 dimenzije

A = eigvec_sort[:,:3]
Y = A.T@X.T

data_LDA_3 = pd.DataFrame(data = Y.T)
data_LDA_3 = pd.concat([data_LDA_3,y],axis = 1)

data_LDA_3.columns = ['LDA1', 'LDA2', 'LDA3', 'COVID19']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color_map = plt.get_cmap('spring')
ax.scatter3D(data_LDA_3.LDA1, data_LDA_3.LDA2, data_LDA_3.LDA3,
             c=data_LDA_3.COVID19, cmap = color_map)
ax.legend()
plt.show()

#%% LDA - 1 dimenzija

A = eigvec_sort[:,:1]
Y = A.T@X.T

data_LDA_1 = pd.DataFrame(data = Y.T)
data_LDA_1 = pd.concat([data_LDA_1,y],axis = 1)

data_LDA_1.columns = ['LDA1', 'COVID19']

plt.figure()
sb.scatterplot(data = data_LDA_1, x = 'LDA1', y = 0 , hue = 'COVID19')
plt.show()

#%% PCA - 2 dimenzije

Sx = np.cov(X.T)

[eigval, eigvec] = np.linalg.eig(Sx)
ind = np.argsort(eigval)[::-1]
eigval = eigval[ind]
eigvec = eigvec[:, ind]

A = eigvec[:, :2]
Y = A.T @ X.T
Y = Y.T
PCA_2 = pd.concat([Y , y] , axis = 1)
PCA_2.columns = ['PCA1', 'PCA2', 'COVID19']

plt.figure()
sb.scatterplot(data = PCA_2, x = 'PCA1',y = 'PCA2' , hue = 'COVID19')
plt.show()

#%% Klasifikacija

data_xy = data_LDA_2.iloc[:,:-1] 
result = data_LDA_2.iloc[:,-1] # COVID-19 DA/NE
K1 = data_xy.loc[result == 0, :] #prva klasa -> zdravi
K2 = data_xy.loc[result == 1, :] #druga klasa -> bolesni
N1 = len(K1)
N2 = len(K2)

x11_ = np.array(K1.LDA1)
x12_ = np.array(K1.LDA2)
x21_ = np.array(K2.LDA1)
x22_ = np.array(K2.LDA2)

N1_training = int(0.7*N1)
N2_training = int(0.7*N2)
N1_test = N1 - N1_training
N2_test = N2 - N2_training

#%% obucavanje/treniranje

x11 = np.zeros((N1_training,1))
x12 = np.zeros((N1_training,1))
x21 = np.zeros((N2_training,1))
x22 = np.zeros((N2_training,1))

for i in range(0,N1_training):
    x11[i] = x11_[i].real
    x12[i] = x12_[i].real

for i in range(0,N2_training):
    x21[i] = x21_[i].real
    x22[i] = x22_[i].real
      
Z1 = np.concatenate((-x11**2, -x11*x12, -x12**2, -x11, -x12,
                     -np.ones((N1_training, 1))), axis=1)
Z2 = np.concatenate((x21**2, x21*x22, x22**2, x21, x22,
                     np.ones((N2_training, 1))), axis=1)
U = np.concatenate((Z1, Z2), axis=0).T

#Gama = np.ones(((N1_training+N2_training), 1)) 
Gama = np.append(1*np.ones((N1_training, 1)), 
                 1*np.ones((N2_training, 1)), axis=0)

W = np.linalg.inv(U@U.T)@U@Gama

V0 = W[-1]
V = np.array([W[3], W[4]])
Q = np.array([[W[0], W[1]], [W[1], W[2]]])
    
xrange = np.linspace(-6.0, 6.0, 100)
yrange = np.linspace(-10.0, 10.0, 100)
x,y = np.meshgrid(xrange,yrange)

equation = V0 + V[0]*x + Q[0,0]*x**2 + 2*Q[0,1]*x*y + Q[1,1]*y**2

plt.figure()
plt.plot(x11, x12, '.', label = 'zdravi')
plt.plot(x21, x22, '.', label = 'bolesni')
plt.legend()
plt.contour(x,y,equation,[0])
plt.show() 

#%% testiranje

x11_t = np.zeros((N1_test,1))
x12_t = np.zeros((N1_test,1))
x21_t = np.zeros((N2_test,1))
x22_t = np.zeros((N2_test,1))

for i in range(0,N1_test):
    x11_t[i] = x11_[N1_training + i].real
    x12_t[i] = x12_[N1_training + i].real

for i in range(0,N2_test):
    x21_t[i] = x21_[N2_training + i].real
    x22_t[i] = x22_[N2_training + i].real
    
decision = np.zeros(((N1_test + N2_test), 1))
conf_mat = np.zeros((2,2))

for i in range(N1_test):
    h = V0 + V[0]*x11_t[i] + Q[0,0]*x11_t[i]**2 + 2*Q[0,1]*x11_t[i]*x12_t[i] 
    + Q[1,1]*x12_t[i]**2
    if h < 0:
        decision[i] = 0
    else:
        decision[i] = 1
        
for i in range(N2_test):
    h = V0 + V[0]*x21_t[i] + Q[0,0]*x21_t[i]**2 + 2*Q[0,1]*x21_t[i]*x22_t[i] 
    + Q[1,1]*x22_t[i]**2
    if h < 0:
        decision[N1_test + i] = 0
    else:
        decision[N1_test + i] = 1
        
#%% konfuziona matrica

#Xtest = np.append(X1_test, X2_test, axis=0)
#Ytest = np.append(np.zeros((400, 1)), np.ones((400, 1)))
Ytest = np.append(np.zeros((N1_test,1)),np.ones((N2_test,1)), axis = 0)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(Ytest, decision)

plt.figure()
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

acc = np.trace(conf_mat)/np.sum(conf_mat)*100
print('Tacnost klasifikatora iznosi:' + str(acc) + '%')

#%% Neuralne mreze 

X = data.drop('COVID-19',axis = 1)
Y = data.iloc[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size = 0.7,
                                                    random_state = 2)

#%% jedan skriveni sloj sa 20 neurona
model = Sequential()
model.add(Dense(20, input_dim = 20, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% jedan skriveni sloj sa 10 neurona 
model = Sequential()
model.add(Dense(10, input_dim = 20, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% jedan skriveni sloj sa 3 neurona
model = Sequential()
model.add(Dense(3, input_dim = 20, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% jedan skriveni sloj sa 500 neurona
model = Sequential()
model.add(Dense(500, input_dim = 20, activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% dva skrivena sloja sa 20 i 15 neurona 

model = Sequential()
model.add(Dense(20, input_dim = 20, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% dva skrivena sloja sa 20 i 10 neurona

model = Sequential()
model.add(Dense(20, input_dim = 20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% dva skrivena sloja sa 20 i 5 neurona

model = Sequential()
model.add(Dense(20, input_dim = 20, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% dva skrivena sloja sa 10 i 3 neurona 

model = Sequential()
model.add(Dense(10, input_dim = 20, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% dva skrivena sloja sa 1000 i 1000 neurona 

model = Sequential()
model.add(Dense(1000, input_dim = 20, activation = 'relu'))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,Y_train,validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% zastita od preobucavanja - regularizacija

model = Sequential()
model.add(Dense(1000, input_dim = 20, activation = 'relu',
                kernel_regularizer=l2(0.0001)))
model.add(Dense(1000, activation = 'relu', 
                kernel_regularizer=l2(0.0001)))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test),
                    epochs = 100, verbose = 0)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

plt.figure()
Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()

#%% zastita od preobucavanja - rano zaustavljanje

model = Sequential()
model.add(Dense(1000, input_dim = 20, activation = 'relu'))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test),
                    callbacks=[es], epochs = 200, verbose = 1)

_, train_acc = model.evaluate(X_train,Y_train, verbose = 0)
print('Train acc = ' + str(train_acc*100) + '%')

_, test_acc = model.evaluate(X_test,Y_test, verbose = 0)
print('Test acc = ' + str(test_acc*100) + '%')

plt.figure()
Y_pom = model.predict(X_test)
Y_pred = 1*(Y_pom > 0.5)
conf_mat = confusion_matrix(Y_test, Y_pred)
sb.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Trening skup", "Validacioni skup"])
plt.title('Kriterijumska funkcija')
plt.show()