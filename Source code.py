#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[5]:


df=web.DataReader('AAPL',data_source='yahoo',start='2012-01-01',end='2019-12-17')


# In[6]:


df


# In[7]:


df.shape


# In[8]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show


# In[9]:


data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil (len(dataset)*.8)
training_data_len


# In[10]:


scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data


# In[11]:


train_data=scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()


# In[12]:


x_train,y_train=np.array(x_train),np.array(y_train)


# In[13]:


x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[14]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[15]:


model.compile(optimizer='adam',loss='mean_squared_error')


# In[16]:


model.fit(x_train,y_train,batch_size=1,epochs=1)


# In[17]:


test_data=scaled_data[training_data_len-60:,:]
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[18]:


x_test=np.array(x_test)


# In[19]:


x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# In[20]:


predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)


# In[22]:


rmse=np.sqrt(np.mean(predictions-y_test)**2)
rmse


# In[23]:


train=data[:training_data_len]
valid=data[training_data_len:]
valid['predictions']=predictions
plt.figure(figsize=(16,8))
plt.title=('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','predictions']])
plt.legend(['Train','Val','predictions'],loc='lower right')
plt.show()


# In[24]:


valid


# In[25]:


apple_quote=web.DataReader('AAPL',data_source='yahoo',start='2012-01-01',end='2019-12-17')
new_df=apple_quote.filter(['Close'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
X_test=[]
X_test.append(last_60_days_scaled)
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
pred_price=model.predict(X_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)


# In[27]:


apple_quote2=web.DataReader('AAPL',data_source='yahoo',start='2012-01-01',end='2019-12-18')
print(apple_quote2['Close'])


# In[43]:


date =input("enter the in the format YYYY-MM-DD")


# In[45]:


apple_quote2=web.DataReader('AAPL',data_source='yahoo',start=date,end=date)
print(apple_quote2['Close'])


# In[ ]:




