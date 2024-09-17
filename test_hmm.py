import numpy as np
import os
from sklearn.cluster import KMeans
import joblib
import pickle

## Load Kmeans model for quantization
M = 80 # number of clusters / emissions classes
kmeans_fn = f'kmeans_model_{M}.pkl'
kmeans_model = joblib.load(f'Kmeans-models/{kmeans_fn}')

## Read in Test data
test_emissions = {} # separate data by .txt file

# TODO: REPLACE DIRECTORY NAME WITH NAME OF TEST DIRECTORY
test_directory = ''

for filename in os.listdir(test_directory):
    filepath = os.path.join(test_directory, filename)
    
    if filename.endswith('.txt'):
        data = np.loadtxt(filepath, delimiter='\t')  
                
        # print(filename + ':')
        # print(data.shape)
        emissions = kmeans_model.predict(data[:, 1:])
        # print(emissions.shape)
        test_emissions[filename] = emissions
        assert np.max(emissions) < M
    else:
        raise Exception('non-.txt file in test folder')

## Get HMM model params for each motion/gesture
S = 20 # number of states in HMM models
test_models_fn = f'HMM-models/model_{M}clusters_{S}states.pkl'
# test_models_fn = 'HMM-models/test_models.pkl'
with open(test_models_fn, 'rb') as f:
    models_dict = pickle.load(f)

# print(models_dict) # remove

## Make predictions on test data

# forward function to get log-prob
def scaled_forward(pi, A, B, O):
    """    
    A: Transition probabilities, shape(S, S), A_ij is prob from i to j
    B: Emission probabilities, shape(S, M)
    pi: Initial state probabilities, shape(S,)
    O: observation sequence, shape(T,)
    """
    # # Base case
    num_observations = len(O)
    num_states = A.shape[0]

    # Initialize forward probabilities matrix
    alpha = np.zeros((num_observations, num_states), dtype=np.float64)
    # Initialize scaling factors
    c = np.zeros(num_observations, dtype=np.float64)
    
    # Base case with scaling
    alpha[0, :] = pi * B[:, O[0]] # make sure to multiply pi by B
    c[0] = 1 / np.sum(alpha[0, :])
    alpha[0, :] *= c[0]

    
    # Recursive step with scaling
    for t in range(1, num_observations):
        # faster version w/ matmul
        new = (alpha[t-1, :].reshape(1, -1) @ A) * B[:, O[t]]
        alpha[t, :] = new.flatten()

        # scaling
        c[t] = 1 / np.sum(alpha[t, :])
        alpha[t, :] *= c[t]
    
    # log probability of the observation sequence using scaling factors, for prediction and checking for convergence
    # Trick on page 273 of Rabiner
    log_prob_observation = -np.sum(np.log(c))
    
    return alpha, c, log_prob_observation

def predict_log_likelihood(A, B, pi, O):
    """
    Predict log-likelihood for scaled forward pass using trick on page 273 of Rabiner
    """
    # Run the forward algorithm with scaling
    _, _, log_likelihood = scaled_forward(A=A, B=B, pi=pi, O=O)
    
    return log_likelihood

predictions_dict = {}

for filename, emissions in test_emissions.items():
    # forward test data through the models
    predictions = [(model_name, predict_log_likelihood(A=model[0],B=model[1], pi=model[2], O=emissions)) for model_name, model in models_dict.items()]
    # convert nan to -inf for sorting
    predictions = [(model_name, float('-inf') if np.isnan(ll) else ll) for model_name, ll in predictions]
    predictions.sort(key= lambda x: -x[1])

    # store top 3 predicitons
    predictions_dict[filename] = predictions[:3]

out_file = "test_hmm_predicitons.txt"
with open(out_file, 'w') as f:
    for model_name, prediction in predictions_dict.items():
        line = f'{model_name}: {prediction}'
        f.write(f'{line}\n')

print(f'Finished testing! Predictions are in {out_file}.')