import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

Q=np.array([.5,.6,.7,.8,.9,.95],dtype=np.float32)

class SQRNet(Model):
    def __init__(self,hidden_units=64,activation="relu",reg=0):
        super().__init__()
        r=l2(reg) if reg>0 else None
        self.h1=Dense(hidden_units,activation=activation,kernel_regularizer=r)
        self.h2=Dense(hidden_units,activation=activation,kernel_regularizer=r)
        self.out=Dense(1,kernel_regularizer=r)
    def call(self,x):
        return self.out(self.h2(self.h1(x)))

class MQRNet(Model):
    def __init__(self,q_values,hidden_units=64,activation="relu",reg=0):
        super().__init__()
        r=l2(reg) if reg>0 else None
        self.h1=Dense(hidden_units,activation=activation,kernel_regularizer=r)
        self.h2=Dense(hidden_units,activation=activation,kernel_regularizer=r)
        self.out=Dense(len(q_values),kernel_regularizer=r)
    def call(self,x):
        return self.out(self.h2(self.h1(x)))

class PMQRNet(MQRNet):
    pass

class MonoNet(Model):
    def __init__(self,q_values,hidden_units=64,activation="relu",reg=0):
        super().__init__()
        r=l2(reg) if reg>0 else None
        self.h1=Dense(hidden_units,activation=activation,kernel_regularizer=r)
        self.h2=Dense(hidden_units,activation=activation,kernel_regularizer=r)
        self.base=Dense(1,kernel_regularizer=r)
        self.delta=Dense(len(q_values)-1,kernel_regularizer=r)
    def call(self,x):
        x=self.h2(self.h1(x)); base=self.base(x); delta=tf.nn.softplus(self.delta(x))
        return tf.concat([base,base+tf.cumsum(delta,axis=1)],axis=1)

def huber_activation(x,delta=2**-8):
    a=tf.abs(x)
    return tf.where(a<=delta,.5*tf.square(x)/delta,a-.5*delta)

class NCMQRNet(Model):
    def __init__(self,q_values,hidden_units=64,activation="sigmoid",reg=0):
        super().__init__()
        r=l2(reg) if reg>0 else None
        self.q_values=q_values
        self.layer1=Dense(hidden_units,activation=activation,kernel_regularizer=r)
        self.layer2=Dense(hidden_units,activation=huber_activation,kernel_regularizer=r)
        self.U=tf.constant(tf.linalg.band_part(tf.ones((hidden_units,hidden_units)),0,-1)[:,-len(q_values):],dtype=tf.float32)
    def call(self,x):
        x=self.layer2(self.layer1(x))
        W=tf.expand_dims(x,-1)*self.U
        return tf.squeeze(tf.matmul(tf.expand_dims(x,1),W),axis=1)

def fit(model_class,xtr,ytr,xv,yv,lam=10,reg=0,lr=1e-3,epochs=300,patience=10,**kwargs):
    def loss(y,p):
        e=tf.reshape(y,(-1,1))-p
        L=tf.reduce_mean(tf.maximum(Q*e,(Q-1)*e))
        return L+(lam*tf.reduce_mean(tf.square(tf.nn.relu(p[:,:-1]-p[:,1:]))) if model_class is PMQRNet else 0)
    def es(): return tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=patience,restore_best_weights=True)
    if model_class is SQRNet:
        ms=[]
        for q in Q:
            m=SQRNet(reg=reg,**kwargs)
            def sloss(y,p,q=q):
                e=tf.reshape(y,(-1,1))-p
                return tf.reduce_mean(tf.maximum(q*e,(q-1)*e))
            m.compile(tf.keras.optimizers.Adam(lr),sloss); m.fit(xtr,ytr,validation_data=(xv,yv),epochs=epochs,batch_size=512,callbacks=[es()],verbose=0); ms.append(m)
        return ms
    m=model_class(Q,reg=reg,**kwargs); m.compile(tf.keras.optimizers.Adam(lr),loss); m.fit(xtr,ytr,validation_data=(xv,yv),epochs=epochs,batch_size=512,callbacks=[es()],verbose=0); return m

def pred(m,x):
    return np.column_stack([a.predict(x,verbose=0).ravel() for a in m]) if isinstance(m,list) else m.predict(x,verbose=0)