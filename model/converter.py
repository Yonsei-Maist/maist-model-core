import tensorflowjs as tfjs

import os


def save_to_js(self, index, path):
    self._model_core.build_model()
    if index > -1:
        self._model_core.model.load_weights(os.path.join(self._base_path,
                                                         './checkpoints/{}_{}.tf'.format(self.name, index)))
    else:
        self._model_core.load_weight()

    tfjs.converters.save_keras_model(self._model_core.model, path)
