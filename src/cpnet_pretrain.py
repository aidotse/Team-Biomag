import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


def _readimgs(ch1, ch2, ch3, values):
    img1 = tf.io.decode_png(tf.io.read_file(ch1),
                            channels=1,
                            dtype=tf.uint16)
    img2 = tf.io.decode_png(tf.io.read_file(ch2),
                            channels=1,
                            dtype=tf.uint16)
    img3 = tf.io.decode_png(tf.io.read_file(ch3),
                            channels=1,
                            dtype=tf.uint16)
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    img3 = tf.image.convert_image_dtype(img3, tf.float32)
    img = tf.concat([img1, img2, img3], axis=2)

    return img,values

"""
class medloss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_med = y_true[y_pred.shape[-1]:]
        y_true = y_true[:y_pred.shape[-1]]
        y_true = tf.cast(y_true, y_pred.dtype)
        y_med = tf.cast(y_med, y_pred.dtype)
        y_pred = y_pred / y_med
        return tf.math.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
"""

def main():
    cpimg_path = Path(sys.argv[1])
    cpfeat_path = Path(sys.argv[2])

    opt = tf.keras.optimizers.SGD(learning_rate=1e-4)
    #opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    modelpath = ""
    #modelpath = "/home/user/nvme/adipocyte/data/crop/models/resnet50v2_epoch_200_valloss_0.33.h5"

    shape = (1024,1024,3)
    epochs = 200
    bs = 6
    val_well = "D04"
    drop_feat = ['Metadata_FoV', 'Metadata_Well' ,'ImageNumber','Count_Defective_lipid_droplets', 'Count_Lipids_no_edge','Count_cells_no_edge', 'Count_nuclei', 'Count_nuclei_no_edge']
    
    imgs_ch1 = [x.as_posix() for x in list(cpimg_path.rglob('*C01_x*.png'))]
    imgs_ch1.sort()
    imgs_ch2 = [x.as_posix() for x in list(cpimg_path.rglob('*C02_x*.png'))]
    imgs_ch2.sort()
    imgs_ch3 = [x.as_posix() for x in list(cpimg_path.rglob('*C03_x*.png'))]
    imgs_ch3.sort()
    df_imgs = pd.DataFrame(data={'ch1': imgs_ch1, 'ch2': imgs_ch2, 'ch3': imgs_ch3})

    gt20x_feat = pd.read_csv(list(cpfeat_path.glob('*20x*.csv'))[0])
    gt20x_feat = gt20x_feat.drop(drop_feat, axis=1)
    gt40x_feat = pd.read_csv(list(cpfeat_path.glob('*40x*.csv'))[0])
    gt40x_feat = gt40x_feat.drop(drop_feat, axis=1)
    gt60x_feat = pd.read_csv(list(cpfeat_path.glob('*60x*.csv'))[0])
    gt60x_feat = gt60x_feat.drop(drop_feat, axis=1)

    # Normalize features separately for each magnitude
    median_20x = gt20x_feat.median().values
    median_40x = gt40x_feat.median().values
    median_60x = gt60x_feat.median().values
    gt20x_feat = gt20x_feat / median_20x
    gt40x_feat = gt40x_feat / median_40x
    gt60x_feat = gt60x_feat / median_60x
    df_feat = pd.concat([gt20x_feat, gt40x_feat, gt60x_feat], ignore_index=True)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    num_feat = df_feat.shape[1]
    df = pd.concat([df_imgs, df_feat], axis=1)
    df = df.dropna()

    df_train = df[~df['ch1'].str.contains(val_well)]
    df_val = df[df['ch1'].str.contains(val_well)]

    """
    median_train = np.zeros((df_train.shape[0], df_train.shape[1]-3), dtype=np.float64)
    median_val = np.zeros((df_val.shape[0], df_val.shape[1]-3), dtype=np.float64)
    median_train[df_train['ch1'].str.contains('20x'), :] = median_20x
    median_val[df_val['ch1'].str.contains('20x'), :] = median_20x
    median_train[df_train['ch1'].str.contains('40x'), :] = median_40x
    median_val[df_val['ch1'].str.contains('40x'), :] = median_40x
    median_train[df_train['ch1'].str.contains('60x'), :] = median_60x
    median_val[df_val['ch1'].str.contains('60x'), :] = median_60x
    """

    ds_train = tf.data.Dataset.from_tensor_slices((df_train['ch1'],
                                                   df_train['ch2'],
                                                   df_train['ch3'],
                                                   df_train.iloc[:,3:]))
    ds_train = ds_train.map(_readimgs)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((df_val['ch1'],
                                                 df_val['ch2'],
                                                 df_val['ch3'],
                                                 df_val.iloc[:,3:]))
    ds_val = ds_val.map(_readimgs)
    ds_val = ds_val.batch(bs)
    ds_val = ds_val.repeat()
    ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    devices = ['/gpu:0', '/gpu:1']
    cd_ops = tf.distribute.HierarchicalCopyAllReduce()
    strategy = tf.distribute.MirroredStrategy(devices=devices,
                                              cross_device_ops=cd_ops)

    with strategy.scope():
        if os.path.exists(modelpath):
            model = tf.keras.models.load_model(modelpath)
        else:
            #base = tf.keras.applications.ResNet50V2(include_top=False,
            #                                        weights=None,
            #                                        input_shape=shape,
            #                                        pooling='avg')
            base = tf.keras.applications.DenseNet121(include_top=False,
                                                     weights=None,
                                                     input_shape=shape,
                                                     pooling='avg')
            dense = tf.keras.layers.Dense(num_feat)(base.output)
            model = tf.keras.models.Model(inputs=base.input, outputs=dense)
            model.compile(optimizer=opt,
                          loss='mae',
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])

    callbacks = []
    checkpoint = tf.keras.callbacks.ModelCheckpoint('models/densenet121_epoch_{epoch:03d}_valloss_{val_loss:.2f}.h5',
                                                    monitor='val_loss')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
    callbacks.append(checkpoint)
    callbacks.append(tensorboard)

    model.fit(ds_train,
              epochs=epochs,
              steps_per_epoch=df_train.shape[0] // bs,
              validation_data=ds_val,
              validation_steps=df_val.shape[0] // bs,
              callbacks=callbacks,
              shuffle=True)

if __name__=="__main__":
    main()
