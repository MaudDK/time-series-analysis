import tensorflow as tf

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        return inputs + delta


def compile_and_fit(model, window, patience=2, max_epochs = 20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer= tf.keras.optimizers.Adam(),
                  metrics = [tf.keras.metrics.MeanAbsoluteError()])
    
    history = model.fit(window.train, 
                        validation_data = window.val, 
                        epochs = max_epochs, callbacks = [early_stopping])
    
    return history