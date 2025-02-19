import pandas as pd
import numpy as np
import os
from scipy.fftpack import dst, dct
from scipy.fft import fft
from scipy.signal import hilbert
import pywt
import json  # Import json for better feature serialization

df = pd.read_csv('C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Yeast_SIPs_All.csv')
pssm_folder = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Yeast_PSSMs'

def pssm(protein_id):
    pssm_file = os.path.join(pssm_folder, f"{protein_id}.parquet")
    return pd.read_parquet(pssm_file).values

def matrix20x20(pssm_matrix):
    return np.dot(pssm_matrix.T, pssm_matrix)

def dst_2d(matrix_20x20):
    transformed_matrix = dst(dst(matrix_20x20, type=2, norm='ortho').T, type=2, norm='ortho').T
    first_row = transformed_matrix[0, :]
    return first_row

def dct_2d(matrix_20x20):
    transformed_matrix = dct(dct(matrix_20x20, type=2, norm='ortho').T, type=2, norm='ortho').T
    first_row = transformed_matrix[0, :]
    return first_row

def wt_2d(matrix_20x20):
    coeffs2 = pywt.dwt2(matrix_20x20, 'haar')
    cA, (cH, cV, cD) = coeffs2
    first_row = cA[0, :]
    return first_row

def fourier_2d(matrix_20x20):
    transformed_matrix = fft(matrix_20x20)
    first_row = transformed_matrix[0, :]
    return np.real(first_row)

def hilbert_2d(matrix_20x20):
    transformed_matrix = hilbert(matrix_20x20)
    first_row = transformed_matrix[0, :]
    return np.real(first_row)

def protein_features(protein_id, transform_type):
    pssm_matrix = pssm(protein_id)
    matrix_20x20 = matrix20x20(pssm_matrix)
    
    if transform_type == 'DST':
        features = dst_2d(matrix_20x20)
    elif transform_type == 'DCT':
        features = dct_2d(matrix_20x20)
    elif transform_type == 'WT':
        features = wt_2d(matrix_20x20)
    elif transform_type == 'Fourier':
        features = fourier_2d(matrix_20x20)
    elif transform_type == 'Hilbert':
        features = hilbert_2d(matrix_20x20)
    else:
        raise ValueError("Unsupported transform type")
    
    return features

# Function to convert features to a comma-separated string
def features_to_string(features):
    return ','.join(map(str, features))

# Transformations
df['Features_A_DST'] = df['Identifier A'].apply(lambda x: features_to_string(protein_features(x, 'DST')))
df['Features_B_DST'] = df['Identifier B'].apply(lambda x: features_to_string(protein_features(x, 'DST')))

df['Features_A_DCT'] = df['Identifier A'].apply(lambda x: features_to_string(protein_features(x, 'DCT')))
df['Features_B_DCT'] = df['Identifier B'].apply(lambda x: features_to_string(protein_features(x, 'DCT')))

df['Features_A_WT'] = df['Identifier A'].apply(lambda x: features_to_string(protein_features(x, 'WT')))
df['Features_B_WT'] = df['Identifier B'].apply(lambda x: features_to_string(protein_features(x, 'WT')))

df['Features_A_Fourier'] = df['Identifier A'].apply(lambda x: features_to_string(protein_features(x, 'Fourier')))
df['Features_B_Fourier'] = df['Identifier B'].apply(lambda x: features_to_string(protein_features(x, 'Fourier')))

df['Features_A_Hilbert'] = df['Identifier A'].apply(lambda x: features_to_string(protein_features(x, 'Hilbert')))
df['Features_B_Hilbert'] = df['Identifier B'].apply(lambda x: features_to_string(protein_features(x, 'Hilbert')))

# Save to CSV files
output_path_dst = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Features/SIP_DST.csv'
output_path_dct = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Features/SIP_DCT.csv'
output_path_wt = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Features/SIP_WT.csv'
output_path_fourier = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Features/SIP_Fourier.csv'
output_path_hilbert = 'C:/Users/nextn/Downloads/GitHub/SIP_PREDICTION_COMPARISON/Datasets/Yeast/Features/SIP_Hilbert.csv'

df[['Identifier A', 'Identifier B', 'Features_A_DST', 'Features_B_DST', 'Interaction']].to_csv(output_path_dst, index=False)
df[['Identifier A', 'Identifier B', 'Features_A_DCT', 'Features_B_DCT', 'Interaction']].to_csv(output_path_dct, index=False)
df[['Identifier A', 'Identifier B', 'Features_A_WT', 'Features_B_WT', 'Interaction']].to_csv(output_path_wt, index=False)
df[['Identifier A', 'Identifier B', 'Features_A_Fourier', 'Features_B_Fourier', 'Interaction']].to_csv(output_path_fourier, index=False)
df[['Identifier A', 'Identifier B', 'Features_A_Hilbert', 'Features_B_Hilbert', 'Interaction']].to_csv(output_path_hilbert, index=False)
