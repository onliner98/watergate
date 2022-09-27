from tensorflow.keras.layers import Input, Conv3D, LeakyReLU, BatchNormalization, Conv3DTranspose, Concatenate
from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import sigmoid, tanh

class UNet3D(Model):
    def __init__(self, feat_channels=16):
        super().__init__()

        self.c1 = Sequential([
        Conv3D(feat_channels//2, 3, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels//2, 3, padding='same'),
        LeakyReLU(alpha=0.2)
        ])

        self.d1 = Sequential([
        Conv3D(feat_channels, 4, strides=2, padding='same'),
        LeakyReLU(alpha=0.2)
        ])

        self.c2 = Sequential([
        Conv3D(feat_channels, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*2, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.d2 = Sequential([
        Conv3D(feat_channels*2, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.c3 = Sequential([
        Conv3D(feat_channels*2, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*4, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.d3 = Sequential([
        Conv3D(feat_channels*4, 4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])


        self.c4 = Sequential([
        Conv3D(feat_channels*4, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*8, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.u1 = Sequential([
        Conv3DTranspose(feat_channels*8, 3, strides=2, padding='same'), #output_padding=1 TODO if it does not work, try this
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*8, 1),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.c5 = Sequential([
        Conv3D(feat_channels*4, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*4, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.u2 = Sequential([
        Conv3DTranspose(feat_channels*4, 3, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*4, 1),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.c6 = Sequential([
        Conv3D(feat_channels*2, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*2, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.u3 = Sequential([
        Conv3DTranspose(feat_channels*2, 3, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels*2, 1),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.c7 = Sequential([
        Conv3D(feat_channels, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2)
        ])

        self.out = Sequential([
        Conv3D(feat_channels, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(feat_channels, 3, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Conv3D(4, 1, padding='same') # 4 output channels for fg_mask, flowx, flowy, flowz
        ])
        
        self.cat = Concatenate()


    def call(self, img, training=False):
        c1 = self.c1(img)
        d1 = self.d1(c1)

        c2 = self.c2(d1)
        d2 = self.d2(c2)

        c3 = self.c3(d2)
        d3 = self.d3(c3)

        c4 = self.c4(d3)

        u1 = self.u1(c4)
        c5 = self.c5(self.cat([u1,c3]))

        u2 = self.u2(c5)
        c6 = self.c6(self.cat([u2,c2]))

        u3 = self.u3(c6)
        c7 = self.c7(self.cat([u3,c1]))

        out = self.out(c7)

        fg_mask = sigmoid(out[...,0:1])
        flow_mask = tanh(out[...,1:])
        mask = self.cat([fg_mask, flow_mask])

        return mask