
from keras.models import Model
from keras import backend as K
from tensorflow.keras.layers import DepthwiseConv2D,Add,BatchNormalization, Dense,Conv2D,GlobalAveragePooling2D,Flatten,Softmax,Activation
from tensorflow.python.keras.engine.input_layer import Input                                 
# 还没有去实现
def ConvMixerLayer(input_layer,dim,kernel_size=9):
    #深度可分离卷 跨链 结构
    x = DepthwiseConv2D(kernel_size=kernel_size,padding='same')(input_layer)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)
    x = Add()([input_layer,x])

    #逐点卷积 结构
    x = Conv2D(filters=dim,kernel_size=1)(x)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)

    return x

def ConvMixer(input_shape=(224,224,3),dim=512,depth=1,kernel_size=9,patch_size=7,n_classes=1000):
    input_img= Input(shape=input_shape) 

    # 第一个卷积块
    x = Conv2D(filters=dim,kernel_size=patch_size,strides=patch_size)(input_img)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)

    # 根据depth 的 ConvMixerLayer 结构
    for _  in range(depth):
        x = ConvMixerLayer(x,dim,kernel_size=kernel_size)

    #全局平局池化
    x =GlobalAveragePooling2D()(x)
    
    x = Flatten()(x)
    x =  Dense(n_classes)(x)
    x = Softmax()(x)

    return Model(input_img,x,name='ConvMixer')

if __name__ == '__main__':
    model = ConvMixer()
    model.summary()