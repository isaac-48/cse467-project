"""PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees," 
International Conference on Learning Representations (ICLR), 2019.
Paper link: https://openreview.net/forum?id=S1zk9iRqF7
Last updated Date: Feburuary 15th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------
main_pategan_experiment.py
- Main function for PATEGAN framework
(1) pategan_main: main function for PATEGAN
"""

# USAGE ::: /opt/homebrew/bin/python3.10 main_pategan_experiment.py --dataset student --input_csv ../Student_data.csv --output_csv ../pategan_synth_out.csv --generate_only --iterations 1 --epsilon 0.2 --k 5 --batch_size 64 --n_s 1 && echo '--- generated preview ---' && head -n 5 ../synthetic_students_pategan_check.csv
# USAGE ::: /opt/homebrew/bin/python3.10 main_pategan_experiment.py --dataset student --input_csv ../Student_data.csv --output_csv ../pategan_synth_out.csv --generate_only --iterations 5 --epsilon 1 --k 10 --batch_size 64 --n_s 5 && echo '--- generated preview ---' && head -n 5 ../synthetic_students_pategan_check.csv 

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_generator import data_generator
from utils import supervised_model_training
from pate_gan import pategan
from sklearn.preprocessing import MinMaxScaler


STUDENT_NUMERIC_PRECISION = {
  'Attendance_Pct': 1,
  'Study_Hours_Per_Day': 1,
  'Previous_GPA': 2,
  'Sleep_Hours': 1,
  'Social_Hours_Week': 0,
  'Final_CGPA': 2
}


def preprocess_csv_data(file_path, drop_columns=None):
  """Read CSV, encode/normalize it, and return metadata for inverse decoding."""

  if drop_columns is None:
    drop_columns = []

  full_data = pd.read_csv(file_path)
  data = full_data.copy()
  input_columns = data.columns.tolist()

  existing_drop_columns = [col for col in drop_columns if col in data.columns]
  dropped_column_values = {}
  for col in existing_drop_columns:
    dropped_column_values[col] = full_data[col].copy()
  if existing_drop_columns:
    data = data.drop(columns=existing_drop_columns)

  categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
  numeric_cols = [col for col in data.columns if col not in categorical_cols]

  categorical_column_map = {}
  for col in categorical_cols:
    categories = [str(v) for v in sorted(data[col].dropna().unique().tolist())]
    categorical_column_map[col] = ['{}_{}'.format(col, cat) for cat in categories]

  integer_like_numeric_cols = []
  for col in numeric_cols:
    col_values = pd.to_numeric(data[col], errors='coerce').dropna().to_numpy()
    if len(col_values) > 0 and np.allclose(col_values, np.round(col_values)):
      integer_like_numeric_cols.append(col)

  numeric_bounds = {}
  for col in numeric_cols:
    numeric_bounds[col] = {
        'min': float(pd.to_numeric(data[col], errors='coerce').min()),
        'max': float(pd.to_numeric(data[col], errors='coerce').max())
    }

  encoded_data = pd.get_dummies(data, drop_first=False)
  encoded_columns = encoded_data.columns.tolist()

  scaler = MinMaxScaler()
  normalized_data = scaler.fit_transform(encoded_data)

  preprocess_info = {
      'input_columns': input_columns,
      'dropped_columns': existing_drop_columns,
      'dropped_column_values': dropped_column_values,
      'original_columns': data.columns.tolist(),
      'encoded_columns': encoded_columns,
      'categorical_cols': categorical_cols,
      'numeric_cols': numeric_cols,
      'categorical_column_map': categorical_column_map,
      'integer_like_numeric_cols': integer_like_numeric_cols,
      'numeric_bounds': numeric_bounds,
      'scaler': scaler
  }

  return normalized_data, preprocess_info


def postprocess_synthetic_data(synth_data, preprocess_info):
  """Convert generated encoded data back to the original table schema."""

  clipped_synth = np.clip(synth_data, 0.0, 1.0)
  decoded_array = preprocess_info['scaler'].inverse_transform(clipped_synth)
  decoded_df = pd.DataFrame(decoded_array, columns=preprocess_info['encoded_columns'])

  output_df = pd.DataFrame(index=np.arange(decoded_df.shape[0]))

  for col in preprocess_info['numeric_cols']:
    col_values = pd.to_numeric(decoded_df[col], errors='coerce')
    bounds = preprocess_info['numeric_bounds'][col]
    col_values = np.clip(col_values, bounds['min'], bounds['max'])

    if col in STUDENT_NUMERIC_PRECISION:
      precision = STUDENT_NUMERIC_PRECISION[col]
      rounded_values = np.round(col_values, precision)
      if precision == 0:
        output_df[col] = rounded_values.astype(int)
      else:
        output_df[col] = rounded_values
    elif col in preprocess_info['integer_like_numeric_cols']:
      output_df[col] = np.round(col_values).astype(int)
    else:
      output_df[col] = col_values

  for col in preprocess_info['categorical_cols']:
    candidate_cols = [
        c for c in preprocess_info['categorical_column_map'][col]
        if c in decoded_df.columns
    ]
    if not candidate_cols:
      output_df[col] = ''
      continue

    top_col = decoded_df[candidate_cols].idxmax(axis=1)
    output_df[col] = top_col.str[len(col) + 1:]

  output_df = output_df[preprocess_info['original_columns']]

  # Re-introduce dropped columns so the final schema matches the input CSV.
  for col in preprocess_info.get('dropped_columns', []):
    source_series = preprocess_info.get('dropped_column_values', {}).get(col, pd.Series(dtype=object))
    source_values = source_series.dropna().to_numpy()
    if len(source_values) == 0:
      output_df[col] = ''
    else:
      output_df[col] = np.random.choice(source_values, size=output_df.shape[0], replace=True)

  if 'input_columns' in preprocess_info:
    output_df = output_df[preprocess_info['input_columns']]

  if 'Student_ID' in output_df.columns:
    output_df = output_df.drop(columns=['Student_ID'])

  return output_df

#%% 
def pategan_main (args):
  """PATEGAN Main function.
  
  Args:
    data_no: number of generated data
    data_dim: number of data dimensions
    noise_rate: noise ratio on data
    iterations: number of iterations for handling initialization randomness
    n_s: the number of student training iterations
    batch_size: the number of batch size for training student and generator
    k: the number of teachers
    epsilon, delta: Differential privacy parameters
    lamda: noise size
    
  Returns:
    - results: performances of Original and Synthetic performances
    - train_data: original data
    - synth_train_data: synthetically generated data
  """
  
  # Supervised model types
  models = ['logisticregression','randomforest', 'gaussiannb','bernoullinb',
            'svmlin', 'Extra Trees','LDA', 'AdaBoost','Bagging','gbm', 'xgb']
  
  # Data generation
  preprocess_info = None
  if args.dataset == 'random':
    train_data, test_data = data_generator(args.data_no, args.data_dim, 
                                           args.noise_rate)
    data_dim = args.data_dim
  elif args.dataset == 'credit':
    # Insert relevant dataset here, and scale between 0 and 1.
    data = pd.read_csv('creditcard.csv').to_numpy()
    data = MinMaxScaler().fit_transform(data)
    train_ratio = 0.5
    train = np.random.rand(data.shape[0])<train_ratio 
    train_data, test_data = data[train], data[~train]
    data_dim = data.shape[1]
  elif args.dataset == 'student':
    drop_columns = [col.strip() for col in args.drop_columns.split(',') if col.strip()]
    data, preprocess_info = preprocess_csv_data(args.input_csv, drop_columns)
    train_data = data
    test_data = None
    data_dim = data.shape[1]
  else:
    raise ValueError('Unsupported dataset: {}'.format(args.dataset))
  
  # Define outputs
  results = None
  
  # Define PATEGAN parameters
  parameters = {'n_s': args.n_s, 'batch_size': args.batch_size, 'k': args.k, 
                'epsilon': args.epsilon, 'delta': args.delta, 
                'lamda': args.lamda}
  
  # Generate synthetic training data
  best_perf = 0.0
  can_score = (not args.generate_only)
  if can_score:
    train_labels = np.unique(np.round(train_data[:, (data_dim - 1)]))
    can_score = can_score and (len(train_labels) <= 2) and (test_data is not None)

  if (not can_score) and (not args.generate_only):
    print('Evaluation disabled: labels are not binary or test split is missing.')
    print('Use --generate_only for data synthesis-only workflow.')

  synth_train_data = None
  
  for it in tqdm(range(args.iterations), desc='PATEGAN Iterations', unit='iter'):
    synth_train_data_temp = pategan(train_data, parameters)
    
    if can_score:
      temp_perf, _ = supervised_model_training(
          synth_train_data_temp[:, :(data_dim-1)], 
          np.round(synth_train_data_temp[:, (data_dim-1)]),
          train_data[:, :(data_dim-1)], 
          np.round(train_data[:, (data_dim-1)]),
          'logisticregression')

      # Select best synthetic data
      if temp_perf > best_perf:
        best_perf = temp_perf.copy()
        synth_train_data = synth_train_data_temp.copy()
    else:
      # In synthesis-only mode, keep the latest sample.
      synth_train_data = synth_train_data_temp.copy()
    
    if can_score:
      tqdm.write('Iteration {}: Best-Perf: {:.4f}'.format(it+1, best_perf))
  
  # Train supervised models
  if can_score:
    results = np.zeros([len(models), 4])
    for model_index in range(len(models)):
      model_name = models[model_index]
      
      # Using original data
      results[model_index, 0], results[model_index, 2] = (
          supervised_model_training(train_data[:, :(data_dim-1)], 
                                    np.round(train_data[:, (data_dim-1)]),
                                    test_data[:, :(data_dim-1)], 
                                    np.round(test_data[:, (data_dim-1)]),
                                    model_name))
          
      # Using synthetic data
      results[model_index, 1], results[model_index, 3] = (
          supervised_model_training(synth_train_data[:, :(data_dim-1)], 
                                    np.round(synth_train_data[:, (data_dim-1)]),
                                    test_data[:, :(data_dim-1)], 
                                    np.round(test_data[:, (data_dim-1)]),
                                    model_name))

    # Print the results for each iteration
    results = pd.DataFrame(np.round(results, 4), 
                           columns=['AUC-Original', 'AUC-Synthetic', 
                                    'APR-Original', 'APR-Synthetic'])
    print(results)
    print('Averages:')
    print(results.mean(axis=0))
  else:
    print('Synthetic data generation complete.')
  
  return results, train_data, synth_train_data, preprocess_info

  
#%%  
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_no',
      help='number of generated data',
      default=10000,
      type=int)
  parser.add_argument(
      '--data_dim',
      help='number of dimensions of generated dimension (if random)',
      default=10,
      type=int)
  parser.add_argument(
      '--dataset',
      help='dataset to use',
      default='random',
      type=str)
  parser.add_argument(
      '--input_csv',
      help='input CSV path for custom datasets such as student',
      default='../Student_data.csv',
      type=str)
  parser.add_argument(
      '--output_csv',
      help='output CSV path for generated synthetic data',
      default='synthetic_students_pategan.csv',
      type=str)
  parser.add_argument(
      '--drop_columns',
      help='comma-separated columns to drop before training',
      default='',
      type=str)
  parser.add_argument(
      '--generate_only',
      help='if set, skip downstream supervised model evaluation',
      action='store_true')
  parser.add_argument(
      '--noise_rate',
      help='noise ratio on data',
      default=1.0,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of iterations for handling initialization randomness',
      default=50,
      type=int)
  parser.add_argument(
      '--n_s',
      help='the number of student training iterations',
      default=1,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of batch size for training student and generator',
      default=64,
      type=int)
  parser.add_argument(
      '--k',
      help='the number of teachers',
      default=10,
      type=int)
  parser.add_argument(
      '--epsilon',
      help='Differential privacy parameters (epsilon)',
      default=1.0,
      type=float)
  parser.add_argument(
      '--delta',
      help='Differential privacy parameters (delta)',
      default=0.00001,
      type=float)
  parser.add_argument(
      '--lamda',
      help='PATE noise size',
      default=1.0,
      type=float)
  
  args = parser.parse_args() 
  
  # Calls main function  
  results, ori_data, synth_data, preprocess_info = pategan_main(args)

  if args.output_csv:
    if (args.dataset == 'student') and (preprocess_info is not None):
      synth_df = postprocess_synthetic_data(synth_data, preprocess_info)
    elif preprocess_info is None:
      synth_df = pd.DataFrame(synth_data)
      if args.input_csv and os.path.exists(args.input_csv):
        input_columns = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
        if len(input_columns) == synth_df.shape[1]:
          synth_df.columns = input_columns
    else:
      synth_df = pd.DataFrame(synth_data, columns=preprocess_info['encoded_columns'])

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    synth_df.to_csv(args.output_csv, index=False)
    print('Saved synthetic data to {}'.format(args.output_csv))