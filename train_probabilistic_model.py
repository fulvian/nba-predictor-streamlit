import os
import sys
sys.path.insert(0, '.')

from probabilistic_model_v2 import ProbabilisticModel

if __name__ == "__main__":
    # Elimina i modelli esistenti
    models_dir = os.path.join('models', 'probabilistic')
    for file in os.listdir(models_dir):
        if file.endswith(".joblib"):
            os.remove(os.path.join(models_dir, file))
    
    # Addestra il modello
    model = ProbabilisticModel()
    model.train_probabilistic_models("training_data_v4.csv")
