#!/usr/bin/env python
# coding: utf-8

# In[1]:


#@import plot
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#@ import Audio
from IPython.display import Audio
from scipy.io import wavfile
from io import BytesIO


# In[3]:


fs=8000
t = np.arange(0,5, step =1./fs)
x = np.sin(2 * np.pi * 586 *  t) 
y =   np.sin(2 * np.pi *  863 *t)
z =x+y
plt.plot(t,z)
plt.xlim(0,.1)


# In[4]:


Audio(z,rate = fs)


# In[5]:


Z=np.fft.fft(z)
Z 


# In[6]:


X_pow = np.abs(Z) **2
plt.plot(X_pow)


# In[7]:


N=len(X_pow)
f_pos =np.arange(0,fs/2,step=fs/N)
plt.plot(f_pos,X_pow[(N//2):])
N,len(f_pos)


# In[8]:


f_neg = np.arange(-fs / 2,0,step=fs/N)
plt.plot(f_neg,X_pow[:(N//2)])


# # FILTER HIGHPASS
#     SKIPING A HIGH FREQUENCY (863) AND OBTAINING A LOW FREQUENCY (586)
# 

# In[9]:


f_pos = np.arange(0, fs/2, step= fs/N)
H_pos = 1. * (f_pos >= 586)
plt.plot(f_pos,H_pos)
plt.ylim(-.1, 1.1)


# In[10]:


f_neg = np.arange( -fs/2, 0, step= fs/N)
H_neg = 1. * (f_neg <= -586)
plt.plot(f_neg,H_neg)
plt.ylim(-.1, 1.1)


# In[11]:


H = np.concatenate([H_pos, H_neg])
plt.plot(H)


# In[12]:


H = np.real(np.fft.fft(H))
plt.plot(H)


# In[13]:


H_trunc = np.concatenate([H[-10000:], H[:10000]])
plt.plot(H_trunc)


# # APPLY THIS FILTER TO SIGNAL
#     USE EVEN NUMBERS TO CHECK THE SIGNAL

# In[14]:


y = np.convolve(z,H_trunc)
y = y[:-1]


# In[19]:


N = len(y)
t = np.arange(0, N/fs, step = 1/fs)
plt.plot(t,y)
plt.xlim(.3,.4)


# In[16]:


f_pos = np.arange(0, fs/2, step = fs/N)
Y = np.fft.fft(y)
Y_pow = np.abs(Y) **2
plt.plot(f_pos, Y_pow[:(N//2)])
plt.xlim(0, 863)


# In[20]:


Audio(y, rate=fs)


# In[ ]:




