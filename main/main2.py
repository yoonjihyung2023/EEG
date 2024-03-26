import pandas as pd
import numpy as np
import spacy
from multiprocessing import Pool
from functools import reduce
import cProfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import FunnelModel, FunnelBaseModel, FunnelTokenizer, FunnelConfig, FunnelModel, AutoTokenizer, BigBirdModel, BigBirdTokenizer, BigBirdConfig, TransfoXLModel, TransfoXLTokenizer, TransfoXLConfig
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss

# Load the dataset (adjust the path as needed)
df1 = pd.read_csv(r'c:\Users\ANDLab10\Desktop\EEG\Datasets\Movielens\genome-scores.csv')
df2 = pd.read_csv(r'c:\Users\ANDLab10\Desktop\EEG\Datasets\Movielens\genome-tags.csv') 
df3 = pd.read_csv(r'c:\Users\ANDLab10\Desktop\EEG\Datasets\Movielens\links.csv') 
df4 = pd.read_csv(r'c:\Users\ANDLab10\Desktop\EEG\Datasets\Movielens\movies.csv')
df5 = pd.read_csv(r'c:\Users\ANDLab10\Desktop\EEG\Datasets\Movielens\ratings.csv')
df6 = pd.read_csv(r'c:\Users\ANDLab10\Desktop\EEG\Datasets\Movielens\tags.csv')

# Concatenating all datasets
concatenated_df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# Assuming 'rating' is the target column
label_column = 'rating'
X = concatenated_df.drop(columns=['rating'], axis=1)
y = concatenated_df['rating']

# Splitting the dataset into training and validation sets
df_train, df_validation = train_test_split(concatenated_df, test_size=0.2, random_state=42)

# Extract labels and apply Label Encoding
train_labels = df_train[label_column]
validation_labels = df_validation[label_column]

# Initialize and fit the Label Encoder
label_encoder = LabelEncoder()
label_encoder.fit(y)

# Apply the Label Encoder
train_labels_encoded = label_encoder.transform(train_labels)
validation_labels_encoded = label_encoder.transform(validation_labels)

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(concatenated_df.drop(label_column, axis=1), concatenated_df[label_column], test_size=0.2, random_state=42)

# Now, X_train, y_train are your training sets
# X_validation, y_validation are your validation sets
# X_test, y_test are your test sets

# Concatenating all datasets
concatenated_df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

# Checking and Selecting Required Columns
required_columns = ['movieId', 'tagId', 'relevance', 'tag', 'imdbId', 'tmdbId', 'title', 'genres', 'userId', 'rating', 'timestamp']
if not set(required_columns).issubset(concatenated_df.columns):
    raise ValueError("Some required columns are missing.")

# Check if all required columns exist in concatenated_df
if all(column in concatenated_df.columns for column in required_columns):
    df_train = concatenated_df[required_columns]
    df_test = concatenated_df[required_columns]
else:
    raise ValueError("Some required columns are missing in the concatenated dataframe.")

# Split your concatenated dataframe into training and validation sets
df_train, df_validation = train_test_split(concatenated_df, test_size=0.2, random_state=42)

# Now you can safely use df_validation
validation_labels = df_validation['rating']

# Data Preparation for Tokenization
tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge-base")
genres = df4['genres'].tolist()
tokenized_genres = tokenizer(genres, padding=True, truncation=True, return_tensors="pt")
labels = df4['genres']  # Replace 'genres' with your actual label column

# Assuming that you have a way to generate or access labels for each title
# Replace 'your_labels_source' with the actual source of your labels
corresponding_labels_after_processing = df4['genres'].tolist()

# Tokenization and Label Encoding Checks
if 'concatenated_df' in globals() and 'genres' in concatenated_df.columns:
    genres = concatenated_df['genres'].dropna().tolist()
    tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge-base")
    tokenized_genres = tokenizer(genres, padding=True, truncation=True, return_tensors="pt")

    if tokenized_genres:
        input_ids_train = tokenized_genres['input_ids']
        attention_masks_train = tokenized_genres['attention_mask']
        print("Tokenization and data extraction complete.")
    else:
        raise ValueError("Tokenization failed. tokenized_genres is None.")
else:
    raise ValueError("DataFrame concatenated_df not defined or 'genres' column missing.")

# Label Encoding and Transformation
label_column = 'rating'  
label_encoder = LabelEncoder()

# Fit the encoder on the combined labels from training and testing data
combined_labels = pd.concat([df_train[label_column], df_test[label_column]], ignore_index=True)
label_encoder.fit(combined_labels)

# Encode the labels in the training and validation sets
train_labels_encoded = label_encoder.transform(df_train[label_column])
validation_labels_encoded = label_encoder.transform(df_validation[label_column])

# If label_column is 'rating' and you have already split y into y_train and y_validation:
train_labels = y_train
validation_labels = y_validation  # Use y_validation directly

# Function to encode labels using the mapping
def encode_labels(labels, encoder):
    """
    Encodes labels using a predefined mapping. Assigns a special value for unseen labels.
    Args:
    - labels: A list or series of labels to encode.
    Returns:
    - A list of encoded label values.
    """
    return encoder.transform(labels)

# Assuming label_column is the name of the label column, label_mapping is your predefined label-to-integer mapping,
# and unseen_label_value is the value to use for unseen labels (e.g., -1 or the max of mapping values + 1)
df_train_encoded = encode_labels(df_train['rating'], label_encoder)
df_test_encoded = encode_labels(df_test['rating'], label_encoder)

# Assuming that you have a way to generate or access labels for each title
corresponding_labels_after_processing = df4['genres'].tolist()

if label_column in df_train.columns and label_column in df_test.columns:
# Initialize and fit the label encoder
   label_encoder = LabelEncoder()
   all_labels = pd.concat([df_train[label_column], df_test[label_column]], ignore_index=True)
   label_encoder.fit(all_labels)

# Assuming you're checking if the label column exists in both datasets
if label_column in df_train.columns and label_column in df_test.columns:
    # Encoding labels
    df_train_encoded = encode_labels(df_train[label_column], label_encoder)
    df_test_encoded = encode_labels(df_test[label_column], label_encoder)
else:
    print("Label column not found in training or testing set.")

# Assuming 'labels' is a list or series of all your labels
unique_labels = set(labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Special value for unseen labels
unseen_label_value = len(label_mapping)

# Function to transform labels with handling of unknown labels
def transform_with_handling_unknowns(data, encoder):
    transformed_data = []
    for label in data:
        try:
            transformed_label = encoder.transform([label])[0]
        except KeyError:
            transformed_label = unseen_label_value
        transformed_data.append(transformed_label)
    return transformed_data

# Converting to PyTorch Tensors
train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), torch.tensor(train_labels))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

validation_data = TensorDataset(torch.tensor(validation_inputs), torch.tensor(validation_masks), torch.tensor(validation_labels))
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=32)

# Creating DataLoaders
batch_size = 32 

# Split tokenized data for training and testing
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids_train, corresponding_labels_after_processing, test_size=0.1, random_state=42)
train_masks, validation_masks, _, _ = train_test_split(attention_masks_train, corresponding_labels_after_processing, test_size=0.1, random_state=42)

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

