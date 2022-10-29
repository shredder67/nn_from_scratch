from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


def construct_mlp(params):
    return MLPRegressor(params)


def get_mlp_trained_weights(model_params, X, y, solver, lr):
    model = construct_mlp(*model_params)
    fitted_model = model.fit(X, y)
    return fitted_model.coefs_