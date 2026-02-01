from sklearn.model_selection import ParameterGrid

input_cont_dim = 120
param_grid = {
    'cont_dim': [input_cont_dim],
    'hidden_dims': [
        [128, 64, 32],      # Original
        [256, 128, 64, 32], # Deeper/Wider
        [64, 32]            # Shallower (prevent overfitting)
    ],
}

parameter_grid = list(ParameterGrid(param_grid))

for i, params in enumerate(parameter_grid):
  print(i)
  name = "_".join(map(str, params['hidden_dims']))
  print(name)
  print("--"*20)