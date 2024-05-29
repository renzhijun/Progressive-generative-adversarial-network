import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model, Sequential
import numpy as np
from functools import partial

# generate noise
def get_random_z(z_dim, batch_size):
    return tf.random.normal([batch_size, z_dim], mean=0., stddev=1)

# define discriminator
def make_discriminaor(input_shape):     
    model = Sequential()
    model.add(layers.Reshape((1, 1024, 1)))
    model.add(layers.Conv2D(32, (1, 5), strides=(1, 4), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, (1, 5), strides=(1, 4), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, (1, 5), strides=(1, 4), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(256, (1, 5), strides=(1, 4), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, use_bias=False))
    signal = layers.Input(shape=input_shape)
    label = model(signal)
        
    return Model(signal, label)


# define generator
def make_generator(input_shape):
    inp1 = layers.Input(shape=(64,))
    inp2 = layers.Input(shape=(1024,))
    out = layers.Dense(1024, use_bias=False)(inp1)
    out = layers.Add()([out, inp2])
    out = layers.Dense(1*16*256, use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Reshape((1, 16, 256))(out)
    out = layers.Conv2DTranspose(128, (1, 5), strides=(1, 4), padding='same', use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2DTranspose(64, (1, 5), strides=(1, 4), padding='same', use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.Conv2DTranspose(32, (1, 5), strides=(1, 4), padding='same', use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out2 = layers.Reshape((1, 1024, 1))(inp2)
    out = layers.Add()([out, out2])
    out = layers.Conv2DTranspose(1, (1, 5), strides=(1, 1), padding='same', use_bias=False)(out)
    out = layers.Reshape((1024,))(out)
    
    return Model(inputs=[inp1, inp2], outputs=out)
    

# Wasserstein Loss
def get_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    def g_loss_fn(fake_logits):
        return -tf.reduce_mean(fake_logits)

    return d_loss_fn, g_loss_fn

# Gradient Penalty (GP)
def gradient_penalty(discriminator, real_images, fake_images, batch_size):
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)
    alpha = tf.random.uniform([batch_size, 1], 0., 1.)
    diff = real_images - fake_images
    inter = fake_images + (alpha * diff)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        predictions = discriminator(inter)
    gradients = tape.gradient(predictions, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, ]))
    return tf.reduce_mean((slopes - 1.) ** 2)

def G3(step, name, time):
    # hyper-parameters
    ITERATION = step #迭代次数
    Z_DIM = 64
    BATCH_SIZE = 2
    BUFFER_SIZE = 8
    G_LR = 0.0004
    D_LR = 0.0004
    GP_WEIGHT = 10.0
    IMAGE_SHAPE = (1024, )
        
    # load sample
    data = np.load('train_sample.npy',allow_pickle=True).item()
    train_x = np.reshape(data[name][8*time:8*(time+1),:,3],(-1,1024)).astype(np.float32)   
    train_ref = np.reshape(data[name][8*time:8*(time+1),:,5],(-1,1024)).astype(np.float32)
    
    # training sample
    train_ds = tf.data.Dataset.from_tensor_slices(train_x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
    train_ref = tf.data.Dataset.from_tensor_slices(train_ref).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
    
    # generator & discriminator
    G = make_generator((Z_DIM,))
    D = make_discriminaor(IMAGE_SHAPE)
    weights_G = np.load('./results/'+name+'_weights/weight_G5_'+str(time)+'.npy', allow_pickle=True)
    weights_D = np.load('./results/'+name+'_weights/weight_D5_'+str(time)+'.npy', allow_pickle=True)
    G.set_weights(weights_G)
    D.set_weights(weights_D)
    
    # optimizer
    g_optim = tf.keras.optimizers.Adam(G_LR, beta_1=0.5, beta_2=0.999)
    d_optim = tf.keras.optimizers.Adam(D_LR, beta_1=0.5, beta_2=0.999)
    
    # loss function
    d_loss_fn, g_loss_fn = get_loss_fn()
        
    # 训练
    @tf.function
    def train_step1(real_images, z, ref_signal):
        with tf.GradientTape() as d_tape1, tf.GradientTape() as g_tape1:
            fake_images = G([z, ref_signal]) + ref_signal
    
            fake_logits = D(fake_images)
            real_logits = D(real_images)
    
            d_loss1 = d_loss_fn(real_logits, fake_logits)
            g_loss1 = g_loss_fn(fake_logits)
    
            gp1 = gradient_penalty(partial(D, training=True), real_images, fake_images, BATCH_SIZE)
            d_loss1 += gp1 * GP_WEIGHT
               
        d_gradients1 = d_tape1.gradient(d_loss1, D.trainable_variables)
        g_gradients1 = g_tape1.gradient(g_loss1, G.trainable_variables)
    
        d_optim.apply_gradients(zip(d_gradients1, D.trainable_variables))
        g_optim.apply_gradients(zip(g_gradients1, G.trainable_variables))
        
        return d_loss1, g_loss1
    # training loop
    def train1(ds1, ds2, L, log_freq=20):  
        ds1 = iter(ds1)
        ds2 = iter(ds2)
        a=[];b=[];
        for step in range(ITERATION):
            noise = np.array(get_random_z(Z_DIM, BATCH_SIZE))
            real_images = next(ds1)
            ref_signal = next(ds2)
            d_loss1, g_loss1 = train_step1(real_images, noise, ref_signal)
            a=np.append(a,d_loss1)
            b=np.append(b,g_loss1)
            if step % log_freq == 0:
                template = '[{}/{}/{}]'
                print(template.format(L, step, ITERATION))
        return a, b
    
    d_loss1, g_loss1 = train1(train_ds, train_ref, 1)
    weights_G = np.array(G.get_weights(),dtype=object)
    weights_D = np.array(D.get_weights(),dtype=object)
    np.save('./results/'+name+'_weights/weight_G3_'+str(time)+'.npy', weights_G)
    np.save('./results/'+name+'_weights/weight_D3_'+str(time)+'.npy', weights_D)
    return d_loss1, g_loss1