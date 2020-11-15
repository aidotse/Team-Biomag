from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


def _readimgs(ch1, ch2, ch3, ch4, values):
    img1 = tf.io.decode_png(tf.io.read_file(ch1),
                            channels=1,
                            dtype=tf.uint16)
    img2 = tf.io.decode_png(tf.io.read_file(ch2),
                            channels=1,
                            dtype=tf.uint16)
    img3 = tf.io.decode_png(tf.io.read_file(ch3),
                            channels=1,
                            dtype=tf.uint16)
    # Normalize 0-65535 to 0-1
    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)
    img3 = tf.image.convert_image_dtype(img3, tf.float32)

    # 60x
    #img1 = tf.image.resize(img1, [512,512])
    #img2 = tf.image.resize(img2, [512,512])
    #img3 = tf.image.resize(img3, [512,512])

    chimg = tf.concat([img1, img2, img3], axis=2)

    imgs = tf.TensorArray(tf.float32, size=7)
    for i in range(7):
        fn = ch4
        if i > 1:
            fn = tf.strings.regex_replace(fn,'Z01','Z{:02d}'.format(i+1))
        img = tf.io.decode_png(tf.io.read_file(fn),
                               channels=1,
                               dtype=tf.uint16)
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 60x
        #img = tf.image.resize(img, [512,512])
        imgs = imgs.write(i, tf.squeeze(img))
    bfimg = imgs.stack()
    bfimg = tf.transpose(bfimg, perm=[1,2,0])

    return bfimg,(chimg,values)


def main():
    val_well = "D04"
    bs = 1
    magnification = "60x"

    cppath = 'models/fullmodel_60x_epoch_{epoch:03d}_valloss_{val_loss:.6f}.h5'
    cpimg_path = Path("/home/user/nvme/adipocyte/data/crop/{}".format(magnification))
    cpfeat_path = Path("/home/user/nvme/adipocyte/data/crop/cellprofiler_features")
    drop_feat = ['Metadata_FoV', 'Metadata_Well' ,'ImageNumber','Count_Defective_lipid_droplets', 'Count_Lipids_no_edge','Count_cells_no_edge', 'Count_nuclei', 'Count_nuclei_no_edge']

    imgs_ch1 = [x.as_posix() for x in list(cpimg_path.glob('*C01_x*.png'))]
    imgs_ch1.sort()
    imgs_ch2 = [x.as_posix() for x in list(cpimg_path.glob('*C02_x*.png'))]
    imgs_ch2.sort()
    imgs_ch3 = [x.as_posix() for x in list(cpimg_path.glob('*C03_x*.png'))]
    imgs_ch3.sort()
    imgs_ch4 = [x.as_posix() for x in list(cpimg_path.glob('*Z01C04_x*.png'))]
    imgs_ch4.sort()

    df_imgs = pd.DataFrame(data={'ch1': imgs_ch1, 'ch2': imgs_ch2, 'ch3': imgs_ch3, 'ch4': imgs_ch4})
    df_imgs = df_imgs[df_imgs['ch1'].str.contains(magnification)]

    gt_feat = pd.read_csv(list(cpfeat_path.glob('*{}*.csv'.format(magnification)))[0])
    gt_feat = gt_feat.drop(drop_feat, axis=1)
    df_feat = gt_feat / gt_feat.median().values
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    num_feat = df_feat.shape[1]
    df = pd.concat([df_imgs, df_feat], axis=1)
    df = df.dropna()
    df_train = df[~df['ch1'].str.contains(val_well)]
    df_val = df[df['ch1'].str.contains(val_well)]

    ds_train = tf.data.Dataset.from_tensor_slices((df_train['ch1'],
                                                   df_train['ch2'],
                                                   df_train['ch3'],
                                                   df_train['ch4'],
                                                   df_train.iloc[:,4:]))
    ds_train = ds_train.map(_readimgs)
    ds_train = ds_train.batch(bs)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((df_val['ch1'],
                                                 df_val['ch2'],
                                                 df_val['ch3'],
                                                 df_val['ch4'],
                                                 df_val.iloc[:,4:]))
    ds_val = ds_val.map(_readimgs)
    ds_val = ds_val.batch(1)
    #ds_val = ds_val.repeat()
    ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    unet = tf.keras.models.load_model('{}_models/unet.h5'.format(magnification))
    cpnet = tf.keras.applications.ResNet50V2(classes=94,
                                             input_shape=(1024,1024,3),
                                             weights='{}_models/cpnet.h5'.format(magnification))(unet.output)
    model = tf.keras.Model(inputs=unet.input, outputs=[unet.output, cpnet])
    sgd = tf.keras.optimizers.SGD(learning_rate=1e-4)
    model.compile(optimizer=sgd,
                  loss='mae',
                  metrics='mae',
                  loss_weights=[10.0, 0.02])
    model.layers[-1].trainable = False

    model.summary(line_length=120)

    callbacks = []
    checkpoint = tf.keras.callbacks.ModelCheckpoint(cppath,
                                                    monitor='val_loss')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
    callbacks.append(checkpoint)
    callbacks.append(tensorboard)

    model.fit(ds_train,
              epochs=40,
              steps_per_epoch=df_train.shape[0] // bs,
              validation_data=ds_val,
              validation_steps=df_val.shape[0] // bs,
              callbacks=callbacks,
              shuffle=True)

    model.save("fullmodel_{}_trained_500x.h5".format(magnification))
    unet.save("unet_{}_trained_500x.h5".format(magnification))


if __name__=="__main__":
    main()
