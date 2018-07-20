
# coding: utf-8

# In[2]:


from tensorflow.contrib.layers import fully_connected
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# <h3>Neural Network Sub-graphs</h3>

# In[3]:


def discriminator(X,inp_shape,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        W_in = tf.Variable(tf.truncated_normal([inp_shape,200],stddev=.1))
        b_in = tf.Variable(tf.truncated_normal([200],stddev=.1))
        
        y_in = tf.matmul(X,W_in) + b_in
        y_in = tf.nn.relu(y_in)

        y_hid = add_layers(200,2,y_in)

        W_out = tf.Variable(tf.truncated_normal([200,1],stddev=.1))
        b_out = tf.Variable(tf.truncated_normal([1],stddev=.1))
        
        logits = tf.matmul(y_hid,W_out) + b_out
        output = tf.sigmoid(logits)

        return output, logits


# In[4]:


def generator(z,inp_shape,out_shape,reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        W_in = tf.Variable(tf.truncated_normal([inp_shape,200],stddev=.1))
        b_in = tf.Variable(tf.truncated_normal([200],stddev=.1))
        
        y_in = tf.matmul(z,W_in) + b_in
        y_in = tf.nn.relu(y_in)

        y_hid = add_layers(200,2,y_in)

        W_out = tf.Variable(tf.truncated_normal([200,out_shape],stddev=.1))
        b_out = tf.Variable(tf.truncated_normal([out_shape],stddev=.1))
        
        y_out = tf.matmul(y_hid,W_out) + b_out
        output = tf.sigmoid(y_out)

        return output


# In[5]:


def dis_pre_proc_img(X_img,reuse=None):
    with tf.variable_scope('img',reuse=reuse):
        X_img = tf.reshape(X_img,[1,224,341,4])
        
        conv1 = convolutional_layer(X_img,shape=[5,5,4,32])
        pool1 = max_pool(conv1)
        
        conv2 = convolutional_layer(conv1,shape=[5,5,32,64])
        pool2 = max_pool(conv2)
        
        conv3 = convolutional_layer(conv2,shape=[5,5,64,128])
        
        flat_img = tf.contrib.layers.flatten(conv3)
        
        return flat_img


# In[6]:


def dis_pre_proc_txt(X_txt,num_inp,reuse=None):
    with tf.variable_scope('txt',reuse=reuse):
        curC = 0.0
        curH = 0.0
        
        for i in range(43):
            curC, curH = small_lstm(num_inp,X_txt[i],curH,curC)
        return curH


# <h3>Manual Neural Network Functions</h3>

# In[7]:


def convolutional_layer(x,shape):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    b = tf.Variable(tf.truncated_normal([shape[3]],stddev=0.1))
    conv = tf.nn.conv2d(x,w,strides=[1,3,3,1],padding='SAME')+b
    act_fun = tf.nn.leaky_relu(conv)
    return act_fun


# In[8]:


def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# In[9]:


def small_lstm(num_inp,x_old,h_old,c_old):
    W_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_c = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    W_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

    U_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_c = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    U_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

    B_f = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    B_i = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    B_o = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))
    B_a = tf.Variable(tf.truncated_normal([num_inp],stddev=.1))

    F_t = tf.sigmoid( tf.multiply(x_old,W_f) + tf.multiply(h_old,U_f) + B_f)
    I_t = tf.sigmoid( tf.multiply(x_old,W_i) + tf.multiply(h_old,U_i) + B_i)
    O_t = tf.sigmoid( tf.multiply(x_old,W_o) + tf.multiply(h_old,U_o) + B_o)
    A_t = tf.tanh(    tf.multiply(x_old,W_a) + tf.multiply(h_old,U_a) + B_a)

    c_t = tf.multiply(I_t,c_old) + tf.multiply(F_t,A_t)
    h_t = tf.multiply(O_t,tf.tanh(c_t))

    return c_t,h_t


# In[10]:


def add_layers(nodes_per_lay,num_lay,lay_1):
    w = tf.Variable(tf.random_uniform([nodes_per_lay,nodes_per_lay]))
    b = tf.Variable(tf.random_uniform([nodes_per_lay]))
    y = tf.nn.relu(tf.matmul(lay_1,w)+b)
    if num_lay == 0:
        return y
    else:
        return add_layers(nodes_per_lay,num_lay-1,y)


# <h3>Data Placeholders</h3>

# In[11]:


txt_inp = tf.placeholder(tf.float32,shape=[None,300])
lat_inp = tf.placeholder(tf.float32,shape=[None,100])
img_inp = tf.placeholder(tf.float32,shape=[None,224,341,4])


# <h3>Generator Text pre-processing</h3>

# In[12]:


txt_vect = dis_pre_proc_txt(txt_inp,300,reuse=None)
print(txt_vect.shape)
txt_vect = tf.reshape(txt_vect,[1,300])
print(txt_vect.shape)


# <h3>Generator Initialization</h3>

# In[13]:


gen_out_shape = 224*341*4
gen_inp_shape = 300+100
gen_inp_vect = tf.concat([lat_inp,txt_vect],axis=1)
G = generator(gen_inp_vect,gen_inp_shape,gen_out_shape,reuse=False)


# <h3>Discriminator Image pre-processing</h3>

# In[14]:


flat_img_shape = 224*341*4
img_vect_real = dis_pre_proc_img(img_inp,reuse=None)
img_vect_fake = dis_pre_proc_img(G,reuse=None)


# <h3>Placeholders</h3>

# In[15]:


real_data = tf.concat([img_vect_real,txt_vect],axis=1)
fake_data = tf.concat([img_vect_fake,txt_vect],axis=1)


# In[16]:


print(img_vect_real.shape)
print(img_vect_fake.shape)
print(txt_vect.shape)
print(real_data.shape)
print(fake_data.shape)


# <h3>Discriminator Initialization</h3>

# In[17]:


D_output_real , D_logits_real = discriminator(real_data,15276)


# In[18]:


D_output_fake, D_logits_fake = discriminator(fake_data,15276,reuse=True)


# <h3>Loss Function</h3>

# In[19]:


def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))


# In[20]:


D_real_loss = loss_func(D_logits_real,tf.ones_like(D_logits_real)* (0.9))


# In[21]:


D_fake_loss = loss_func(D_logits_fake,tf.zeros_like(D_logits_real))


# In[22]:


D_loss = D_real_loss + D_fake_loss


# In[23]:


I_loss = D_loss


# In[24]:


T_loss = D_loss


# In[25]:


G_loss = loss_func(D_logits_fake,tf.ones_like(D_logits_fake))


# <h3>Optimizers</h3>

# In[26]:


learning_rate = 0.001


# In[27]:


tvars = tf.trainable_variables()

i_vars = [var for var in tvars if 'img' in var.name]
t_vars = [var for var in tvars if 'txt' in var.name]
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]


# In[28]:


I_trainer = tf.train.AdamOptimizer(learning_rate).minimize(I_loss, var_list=i_vars)
T_trainer = tf.train.AdamOptimizer(learning_rate).minimize(T_loss, var_list=t_vars)
D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)


# <h3>Training</h3>

# In[29]:


batch_size = 1
epochs = 2
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)


# In[30]:


samples = []
data = np.load('data.npy')


# In[31]:


print(data.shape)
print(data[0].shape)
print(data[0][0].shape)
print(data[0][1].shape)


# In[ ]:


with tf.Session() as sess:
    sess.run(init)
    
    # Recall an epoch is an entire run through the training data
    for e in range(epochs):
         #// indicates classic division
        #num_batches = mnist.train.num_examples // batch_size
        num_batches = data.shape[0]
        for i in range(num_batches):
            #img_inp = tf.placeholder(tf.float32,shape=[None,None])
            #txt_inp = tf.placeholder(tf.float32,shape=[None,300])
            #lat_inp = tf.placeholder(tf.float32,shape=[None,100])
            

            #real_data = tf.concat([img_vect_real,txt_vect],axis=1)
            #fake_data = tf.concat([img_vect_fake,txt_vect],axis=1)
            
            # Grab batch of images
            batch_img = np.reshape(data[i][0],(1,224, 341, 4))
            batch_img = np.reshape
            batch_txt = data[i][1]
            batch_lat = np.random.uniform(-1, 1, size=(1, 100))
            
            # Run optimizers, no need to save outputs, we won't use them
            _ = sess.run(I_trainer, feed_dict={img_inp: batch_img, txt_inp: batch_txt, lat_inp: batch_lat})
            _ = sess.run(T_trainer, feed_dict={img_inp: batch_img, txt_inp: batch_txt, lat_inp: batch_lat})  
            _ = sess.run(D_trainer, feed_dict={img_inp: batch_img, txt_inp: batch_txt, lat_inp: batch_lat})
            _ = sess.run(G_trainer, feed_dict={lat_inp: batch_lat, txt_inp: batch_txt})
      
            
        print("Currently on Epoch {} of {} total...".format(e+1, epochs))
        
        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        batch_txt = data[5][0]
        gen_sample = sess.run(generator(z ,reuse=True),feed_dict={lat_inp: sample_z, txt_inp: batch_txt})
        
        samples.append(gen_sample)

