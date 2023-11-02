from tensorflow.keras.models import model_from_json


class KerasModelLoader:
    _model = None

    @classmethod
    def load_model(cls, model_json_path, model_weights_path):
        if cls._model is None:
            with open(model_json_path, 'r') as json_file:
                model_json = json_file.read()
                cls._model = model_from_json(model_json)
            cls._model.load_weights(model_weights_path)

    @classmethod
    def get_model(cls):
        if cls._model is None:
            raise Exception("Model is not loaded. Please call load_model first.")
        return cls._model