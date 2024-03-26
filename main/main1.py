import pandas as pd
import numpy as np
import spacy
from multiprocessing import Pool
import cProfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import FunnelBaseModel, FunnelTokenizer, FunnelConfig, FunnelModel, AutoTokenizer, BigBirdModel, BigBirdTokenizer, BigBirdConfig, TransfoXLModel, TransfoXLTokenizer, TransfoXLConfig
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read CSV with optimization 
def read_csv_optimized(file_path, chunk_size=50000):  # Further reduced chunk size
    chunks = pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)
    optimized_chunks = []
    for chunk in chunks:
        # Optimize data types here
        # For example, convert object columns to category if applicable
        # chunk['some_column'] = chunk['some_column'].astype('category')
        optimized_chunks.append(chunk)
    return pd.concat(optimized_chunks, ignore_index=True)

# Preprocess Data
def preprocess_data(file_path1, file_path2):
    try:
        data1 = pd.read_csv(file_path1)
        data2 = pd.read_csv(file_path2)
        merged_data = pd.merge(data1, data2, on='Title') 
        return merged_data
    except Exception as e:
        print(f"Error during data processing: {e}")
        return None

# Use this function to load your data
file1 = r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_data.csv'
file2 = r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_rating.csv'
merged_data = preprocess_data(file1, file2)

if merged_data is None:
    print("merged_data is None. Check data loading and processing steps.")
else:
    print("merged_data is ready for further processing.")

# Checking if merged_data is not None before attempting the conversion
if merged_data is not None:
    float_cols = merged_data.select_dtypes(include='float64').columns

    # Check if there are any float64 columns to convert
    if not float_cols.empty:
        merged_data[float_cols] = merged_data[float_cols].astype('float32')
        print("Float columns converted to float32.")
    else:
        print("No float64 columns found to convert.")
else:
    print("No data to process. merged_data is None.")

# Checking whether merged_data is not None and whether each column in columns_to_convert exists in merged_data before attempting the conversion
if merged_data is not None:
    columns_to_convert = ['ratingsCount', 'Price', 'review/score', 'review/time']
    missing_cols = [col for col in columns_to_convert if col not in merged_data.columns]

    if not missing_cols:
        for col in columns_to_convert:
            # Remove commas and convert to float
            merged_data[col] = merged_data[col].astype(str).str.replace(',', '').astype(float)
        print("Columns converted successfully.")
    else:
        print(f"Missing columns in DataFrame: {missing_cols}")
else:
    print("No data to process. merged_data is None.")

# Label encoding the 'Title' column in merged_data 
if merged_data is not None:
    if 'Title' in merged_data.columns:
        label_encoder = LabelEncoder()
        merged_data['Title'] = label_encoder.fit_transform(merged_data['Title'])
        print("Column 'Title' encoded successfully.")
    else:
        print("Column 'Title' not found in DataFrame.")
else:
    print("No data to process. merged_data is None.")

# Applying OneHotEncoding to specific columns in merged_data 
if merged_data is not None:
    low_cardinality_cols = ['ratingsCount', 'Price', 'review/score', 'review/time']

    # Check if all specified columns exist in the DataFrame
    missing_cols = [col for col in low_cardinality_cols if col not in merged_data.columns]
    if not missing_cols:
        encoder = OneHotEncoder()
        encoded_columns = encoder.fit_transform(merged_data[low_cardinality_cols])
        encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_columns, columns=encoder.get_feature_names_out(low_cardinality_cols))
        print("OneHotEncoding applied successfully.")
    else:
        print(f"Missing columns in DataFrame: {missing_cols}")
else:
    print("No data to process. merged_data is None.")

# Normalizing the numerical columns in merged_data using StandardScaler 
if merged_data is not None:
    numerical_cols = ['ratingsCount', 'Price', 'review/score', 'review/time']

    # Check if all numerical columns exist in the DataFrame
    missing_cols = [col for col in numerical_cols if col not in merged_data.columns]
    if not missing_cols:
        scaler = StandardScaler()
        merged_data[numerical_cols] = scaler.fit_transform(merged_data[numerical_cols])
        print("Numerical columns normalized using StandardScaler.")
    else:
        print(f"Missing columns in DataFrame: {missing_cols}")
else:
    print("No data to process. merged_data is None.")

# Handling string columns in merged_data
if merged_data is not None:
    string_columns = ['description']
    
    # Check if all specified columns exist in the DataFrame
    missing_cols = [col for col in string_columns if col not in merged_data.columns]
    if not missing_cols:
        for col in string_columns:
            # Fill missing values with empty strings and ensure the column is of string type
            merged_data[col] = merged_data[col].fillna('').astype(str)
        print("String columns processed successfully.")
    else:
        print(f"Missing columns in DataFrame: {missing_cols}")
else:
    print("No data to process. merged_data is None.")

# Modular function for checking and concatenating text columns
def concatenate_text_columns(df, columns_to_concat):
    missing_cols = [col for col in columns_to_concat if col not in df.columns]
    if not missing_cols:
        concatenated_text = df[columns_to_concat].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        return concatenated_text, None
    else:
        return None, missing_cols

# Modular function for tokenizing text using spaCy
def tokenize_with_spacy(texts, model='en_core_web_sm'):
    nlp = spacy.load(model)
    with Pool(processes=4) as pool:
        tokenized_texts = pool.map(lambda text: [token.text for token in nlp(text)], texts)
    return tokenized_texts

# Modular function for tokenizing text using Funnel Transformer
def tokenize_with_funnel(texts, model="funnel-transformer/xlarge-base"):
    try:
        tokenizer = FunnelTokenizer.from_pretrained(model)
        tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        return tokenized_data, None
    except Exception as e:
        return None, str(e)

# Main script
def process_data(merged_data):
    if merged_data is not None:
        columns_to_concat = ['description']
        
        concatenated_text, missing_cols = concatenate_text_columns(merged_data, columns_to_concat)
        if concatenated_text is not None:
            merged_data['combined_text'] = concatenated_text

            # Tokenization with spaCy
            spacy_tokenized_texts = tokenize_with_spacy(merged_data['combined_text'].tolist())

            # Tokenization with Funnel Transformer
            funnel_tokenized_data, error_message = tokenize_with_funnel(merged_data['combined_text'].tolist())
            if error_message:
                print(f"Error in Funnel Transformer tokenization: {error_message}")
        else:
            print(f"Missing columns in DataFrame: {missing_cols}")
    else:
        print("No data to process. merged_data is None.")

# Optionally delete the combined_text column to save memory
if merged_data is not None:
    # Check if 'combined_text' column exists in merged_data
    if 'combined_text' in merged_data.columns:
        del merged_data['combined_text']
    else:
        print("'combined_text' column does not exist in the DataFrame.")
else:
    print("No data to modify. merged_data is None.")

# Define the label column for your CTR Prediction task
label_column = 'ratingsCount'  

# Tokenization process (assuming you have text data to tokenize)
def tokenize_data(df, text_column, tokenizer_model="funnel-transformer/xlarge-base", max_length=512):
    tokenizer = FunnelTokenizer.from_pretrained(model)
    tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    return tokenized_outputs

# Dataset and Dataloader Creation
def create_dataset(merged_data, text_column):
    if merged_data is not None:
        if 'ratingsCount' in merged_data.columns and label_column in merged_data.columns:
            # Creating tensors for labels and features
            labels = torch.tensor(merged_data['ratingsCount'].values, dtype=torch.long)
            tokenized_data = tokenize_data(merged_data, text_column)
            input_ids = tokenized_data['input_ids']
            attention_mask = tokenized_data['attention_mask']
            target_labels = torch.tensor(merged_data[label_column].values, dtype=torch.long)

            # Creating the dataset
            dataset = TensorDataset(input_ids, attention_mask, target_labels)
            return dataset
        else:
            print("Missing required columns in merged_data.")
            return None
    else:
        print("No data to create dataset. merged_data is None.")
        return None

# Load pre-trained models
def load_pretrained_model(model_name):
    if model_name == "funnel-transformer/xlarge-base":
        return FunnelBaseModel.from_pretrained(model_name)
    # Add other models as necessary
    raise ValueError("Unsupported model name")

# Dictionary of model classes and their corresponding configuration classes
def initialize_model(model_name, config=None):
    if model_name == "funnel-transformer/xlarge-base":
        return FunnelModel.from_pretrained(model_name, config=config if config else None)
    raise ValueError("Unsupported model name")

# Initialize the configuration for the Funnel model
funnel_config = FunnelConfig()

# Load the Funnel model with the specified configuration
model = FunnelModel.from_pretrained("funnel-transformer/xlarge-base", config=funnel_config, ignore_mismatched_sizes=True)
# model will be used in training

# Define the model name
model_name = "funnel-transformer/xlarge-base"

# Initialize the base model with the configuration
base_model = FunnelBaseModel.from_pretrained("funnel-transformer/xlarge-base")

# Assuming initialize_model is a function you have defined
# Initialize the funnel model
funnel_model = initialize_model(model_name)

# Incorporating Memory Augmentated Transformer Model for CTR Prediction
class MemoryAugmentedCTRTransformer(nn.Module):
    def __init__(self, base_model, memory_size):
        super().__init__()
        self.base_model = base_model
        if not hasattr(self.base_model, 'config'):
            raise AttributeError(f"The provided base model of type '{type(self.base_model).__name__}' does not have a 'config' attribute. Please check the model initialization.")
        
        self.memory = nn.Parameter(torch.randn(memory_size, self.base_model.config.hidden_size))
        self.fc = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, x):
        x = self.base_model(x, return_dict=True).last_hidden_state[:, 0]  # Extract CLS embedding
        x = x + self.memory
        return torch.sigmoid(self.fc(x))

# Instantiate MemoryAugmentedCTRTransformer
funnel_model_name = "funnel-transformer/xlarge-base"
memory_size = 100
memory_augmented_funnel = MemoryAugmentedCTRTransformer(funnel_model, memory_size=100)

# Initializing the Model
def initialize_model(model_name, memory_size=100, custom_config=None):
    # Dictionary of model classes and their corresponding configuration classes
    model_classes = {
        "funnel-transformer/xlarge-base": (FunnelModel, FunnelConfig),
        "google/bigbird-roberta-base": (BigBirdModel, BigBirdConfig),
        "transfo-xl": (TransfoXLModel, TransfoXLConfig),
        # Add more models here as needed
    }

    if model_name not in model_classes:
        raise ValueError(f"Model '{model_name}' not supported.")

    ModelClass, ConfigClass = model_classes[model_name]
    
    # Use custom configuration if provided, else default configuration
    config = custom_config if custom_config else ConfigClass()

    # Initialize the base model with the configuration
    base_model = ModelClass.from_pretrained(model_name, config=config)
    
    # Wrap the base model in the MemoryAugmentedCTRTransformer
    augmented_model = MemoryAugmentedCTRTransformer(base_model, memory_size)
    return augmented_model

# Define the configuration for the FunnelModel
funnel_config = FunnelConfig(
    vocab_size=30522,  # Adjust as needed for your dataset
    d_model=768,       # Model dimension
    n_head=12,         # Number of attention heads
    d_ff=3072,         # Dimension of the feed-forward layer
    block_sizes=[6, 6],  # A list indicating the number of layers in each block
    # ... other necessary configuration parameters
)

# Create an instance of the FunnelModel with this configuration
funnel_model = FunnelModel(config=funnel_config)
memory_augmented_funnel = MemoryAugmentedCTRTransformer(funnel_model, memory_size=100) 

# Defining num_heads
num_heads = 8  # The number of attention heads in the Transformer model
num_layers = 4
dropout_rate = 0.1
feature_size = 1000
memory_size = 100

# Funnel Transformer with Memory Augmentation
funnel_config = FunnelConfig()
funnel_model = FunnelModel(funnel_config)
memory_augmented_funnel = MemoryAugmentedCTRTransformer(funnel_model, memory_size)

# Transformer XL with Memory Augmentation
TransfoXL_config = TransfoXLConfig()
TransfoXL_model = TransfoXLModel(TransfoXL_config)
memory_augmented_transfoxl = MemoryAugmentedCTRTransformer(TransfoXL_model, memory_size)

# BigBird with Memory Augmentation
bigbird_config = BigBirdConfig(block_size=1024, num_random_blocks=3)
model_name = "google/bigbird-roberta-base"
BigBirdModel = BigBirdModel.from_pretrained(model_name, config=bigbird_config)
memory_augmented_BigBirdModel = MemoryAugmentedCTRTransformer(BigBirdModel, memory_size)

# Define the Multi-Head Attention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (self.head_dim * num_heads == embed_size), "Embed size must be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Implement the multi-head attention forward pass
        return out

# Define the Funnel Attention Block
class FunnelAttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads, pool_size, dropout_rate, forward_expansion):
        super(FunnelAttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transition = nn.Linear(embed_size, forward_expansion * embed_size)
        
        self.pool = nn.AvgPool1d(pool_size, stride=pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm(attention + query))
        x = self.transition(x)
        x = x.permute(0, 2, 1)  # Permute for correct pooling dimensions
        pooled_x = self.pool(x)
        pooled_x = pooled_x.permute(0, 2, 1)  # Permute back to original dimensions
        return pooled_x

# Define a Funnel Transformer Layer
class FunnelTransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, pool_size, dropout_rate, forward_expansion):
        super(FunnelTransformerLayer, self).__init__()
        self.layers = nn.ModuleList([
            FunnelAttentionBlock(embed_size, num_heads, pool_size, dropout_rate, forward_expansion) 
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

# Define Funnel Transformer Class
class FunnelCTRTransformer(nn.Module):
    def __init__(self, feature_size, d_model, nhead, num_layers, dropout_rate):
        super().__init__()
        # Embedding and initial layers
        self.embed = nn.Embedding(feature_size, d_model)
        self.funnel_blocks = nn.ModuleList()
        for i in range(num_layers):
            is_downsampling = (i % 2 == 0)  # Example: downsample every alternate layer
            self.funnel_blocks.append(FunnelAttentionBlock(d_model, nhead, dropout_rate, is_downsampling))
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        for block in self.funnel_blocks:
            x = block(x)
        # Classifier
        x = x.mean(dim=1)  # Pool over sequence
        x = torch.sigmoid(self.fc(x))
        return x

# BigBird Attention Block
class BigBirdAttention(nn.Module):
    def __init__(self, d_model, nhead, num_global_tokens, num_random_tokens, window_size, dropout_rate):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate)
        self.num_global_tokens = num_global_tokens
        self.num_random_tokens = num_random_tokens
        self.window_size = window_size

    def forward(self, x):
        # Implementing simplified attention mechanism
        attention_output = self.self_attention(x, x, x)[0]
        return attention_output

# Define BigBird Block
class BigBirdBlock(nn.Module):
    def __init__(self, d_model, nhead, num_global_tokens, num_random_tokens, window_size, dropout_rate):
        super().__init__()
        self.attention = BigBirdAttention(d_model, nhead, num_global_tokens, num_random_tokens, window_size, dropout_rate)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.norm(x + self.attention(x))
        x = self.norm(x + F.relu(self.ff(x)))
        return x

# Define BigBird Transformer Class
class BigBirdCTRTransformer(nn.Module):
    def __init__(self, feature_size, d_model, nhead, num_layers, num_global_tokens, num_random_tokens, window_size, dropout_rate):
        super().__init__()
        self.embed = nn.Embedding(feature_size, d_model)
        self.layers = nn.ModuleList([BigBirdBlock(d_model, nhead, num_global_tokens, num_random_tokens, window_size, dropout_rate) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)  # Pool over sequence
        x = torch.sigmoid(self.fc(x))
        return x

# Define Transformer XL Layer
class TransformerXLLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout_rate):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate)
        self.feed_forward = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, memory):
        # Concatenate memory (hidden states from previous segment)
        x_with_memory = torch.cat([memory, x], dim=0)
        attention_output, _ = self.self_attention(x_with_memory, x_with_memory, x_with_memory)
        x = self.norm1(x + self.dropout(attention_output[-len(x):]))
        x = self.norm2(x + self.dropout(F.relu(self.feed_forward(x))))
        return x

# Define Transformer XL Class
class TransformerXLModel(nn.Module):
    def __init__(self, num_features, embedding_size, d_model, nhead, num_layers, dropout_rate, memory_size):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_size)
        self.layers = nn.ModuleList([TransformerXLLayer(d_model, nhead, dropout_rate) for _ in range(num_layers)])
        self.fc = nn.Linear(embedding_size, 1)
        self.memory_size = memory_size

    def _reset_memory(self):
        return torch.zeros(self.memory_size, dtype=torch.float)

    def forward(self, x):
        x = self.embedding(x)
        memory = self._reset_memory()

        for layer in self.layers:
            x = layer(x, memory)
            memory = x.detach()

        x = torch.sigmoid(self.fc(x.mean(dim=1)))
        return x

# Define AutoInt Model
class InteractingLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size=embedding_size, num_heads=num_heads, dropout=dropout_rate)

    def forward(self, x):
        x = x.transpose(0, 1)
        out, _ = self.attention(x, x, x)
        return out.transpose(0, 1)

class AutoIntModel(nn.Module):
    def __init__(self, num_features, embedding_size, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_size)
        self.interacting_layers = nn.ModuleList([InteractingLayer(embedding_size, num_heads, dropout_rate) for _ in range(num_layers)])
        self.fc = nn.Linear(num_features * embedding_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.interacting_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        x = torch.sigmoid(self.fc(x))
        return x

# Defining DIN(Deep Interest Network) Model
class AttentionLayer(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, query, keys):
        attention_scores = self.fc(torch.tanh(query + keys))
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(attention_weights * keys, dim=1)
        return weighted_sum

class DINModel(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_size)
        self.attention_layer = AttentionLayer(embedding_size)
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, user_query, user_behaviors):
        user_query_embedded = self.embedding(user_query)
        user_behaviors_embedded = self.embedding(user_behaviors)
        user_interest = self.attention_layer(user_query_embedded, user_behaviors_embedded)
        x = torch.sigmoid(self.fc(user_interest))
        return x

# Defining DIEN(Deep Interest Evolution Network) Model
class InterestEvolutionLayer(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.attention_layer = AttentionLayer(embedding_size)

    def forward(self, user_behaviors):
        gru_out, _ = self.gru(user_behaviors)
        return gru_out

class DIENModel(nn.Module):
    def __init__(self, num_features, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_size)
        self.interest_evolution_layer = InterestEvolutionLayer(embedding_size)
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, user_query, user_behaviors):
        user_query_embedded = self.embedding(user_query)
        user_behaviors_embedded = self.embedding(user_behaviors)
        evolved_interests = self.interest_evolution_layer(user_behaviors_embedded)
        user_interest = self.attention_layer(user_query_embedded, evolved_interests)
        x = torch.sigmoid(self.fc(user_interest))
        return x

# Data Loading and Preprocessing
def preprocess_data(data_paths):
    data1 = pd.read_csv(data_paths[0], low_memory=False)
    data2 = pd.read_csv(data_paths[1], low_memory=False)
    merged_data = pd.merge(data1, data2, on='Title')
    merged_data.drop_duplicates(inplace=True)
    merged_data.ffill(inplace=True)
    return merged_data

# Load the data
merged_data = preprocess_data([r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_data.csv', 
                               r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_rating.csv'])

# Modular function for checking and concatenating text columns
def concatenate_text_columns(df, columns_to_concat):
    missing_cols = [col for col in columns_to_concat if col not in df.columns]
    if not missing_cols:
        concatenated_text = df[columns_to_concat].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        return concatenated_text, None
    else:
        return None, missing_cols

# Modular function for tokenizing text using spaCy
def tokenize_with_spacy(texts, model='en_core_web_sm'):
    nlp = spacy.load(model)
    with Pool(processes=4) as pool:
        tokenized_texts = pool.map(lambda text: [token.text for token in nlp(text)], texts)
    return tokenized_texts

# Modular function for tokenizing text using Funnel Transformer
def tokenize_with_funnel(texts, model="funnel-transformer/xlarge-base"):
    try:
        tokenizer = FunnelTokenizer.from_pretrained(model)
        tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        return tokenized_data, None
    except Exception as e:
        return None, str(e)

# Main script
def process_data(merged_data):
    if merged_data is not None:
        columns_to_concat = ['description']
        
        concatenated_text, missing_cols = concatenate_text_columns(merged_data, columns_to_concat)
        if concatenated_text is not None:
            merged_data['combined_text'] = concatenated_text

            # Tokenization with spaCy
            spacy_tokenized_texts = tokenize_with_spacy(merged_data['combined_text'].tolist())

            # Tokenization with Funnel Transformer
            funnel_tokenized_data, error_message = tokenize_with_funnel(merged_data['combined_text'].tolist())
            if error_message:
                print(f"Error in Funnel Transformer tokenization: {error_message}")
        else:
            print(f"Missing columns in DataFrame: {missing_cols}")
    else:
        print("No data to process. merged_data is None.")

# Define the label column for your CTR Prediction task
label_column = 'ratingsCount'  

# Tokenization process (assuming you have text data to tokenize)
def tokenize_data(df, text_column, tokenizer_model="funnel-transformer/xlarge-base", max_length=512):
    tokenizer = FunnelTokenizer.from_pretrained(model)
    tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    return tokenized_outputs

# Dataset and Dataloader Creation
def create_dataset(merged_data, text_column):
    if merged_data is not None:
        if 'ratingsCount' in merged_data.columns and label_column in merged_data.columns:
            # Creating tensors for labels and features
            labels = torch.tensor(merged_data['ratingsCount'].values, dtype=torch.long)
            tokenized_data = tokenize_data(merged_data, text_column)
            input_ids = tokenized_data['input_ids']
            attention_mask = tokenized_data['attention_mask']
            target_labels = torch.tensor(merged_data[label_column].values, dtype=torch.long)

            # Creating the dataset
            dataset = TensorDataset(input_ids, attention_mask, target_labels)
            return dataset
        else:
            print("Missing required columns in merged_data.")
            return None
    else:
        print("No data to create dataset. merged_data is None.")
        return None

# Paths to CSV files
data_path = r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_data.csv'
rating_path = r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_rating.csv'

# Load the data into DataFrames
data_df = pd.read_csv(data_path)
rating_df = pd.read_csv(rating_path)

# Identify the common column name
common_column_name = 'Title'

# Merge the DataFrames on the common column
merged_df = pd.merge(data_df, rating_df, on=common_column_name)

# Extract the relevant text columns from merged_df
# Ensure that these column names match exactly with those in your merged_df
texts = merged_df[['description']].astype(str).agg(' '.join, axis=1).tolist()

# Initialize the tokenizer
tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge-base")

# Tokenize the texts
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

# Define your targets here using merged_df
target_labels = torch.tensor(merged_df['ratingsCount'].values)

# Define dataset class
class TensorDataset(Dataset):
    def __init__(self, input_ids, attention_mask, target_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.target_labels = target_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'target_labels': self.target_labels[idx]
        }

# Function to create a DataLoader
def create_dataloader(dataset, batch_size, is_train=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)

# Split the dataset into training and validation sets
dataset = TensorDataset(input_ids, attention_mask, target_labels)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model and Training Setup
def setup_training(model, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    return model, optimizer, criterion

model, optimizer, criterion = setup_training(memory_augmented_funnel)

# Class dataset
class TensorDataset(Dataset):
    def __init__(self, input_ids, attention_mask, target_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.target_labels = target_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'target_labels': self.target_labels[idx]
        }

# Function to create a DataLoader
def create_dataloader(dataset, batch_size, is_train=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)

# Read CSV files and process them into tensors or numpy arrays. And, Process data_df to extract input_ids, attention_mask, target_labels
data_df = pd.read_csv(r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_data.csv')
rating_df = pd.read_csv(r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_rating.csv')

# Initialize the tokenizer
tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

# Check the first few entries in the 'description' column
print(data_df['description'].head())

# Check for any non-string or missing values
if not all(isinstance(description, str) for description in data_df['description']):
    print("Non-string values found in 'description' column. Cleaning data...")

    # Replace non-string and NaN values with empty strings
    data_df['description'] = data_df['description'].apply(lambda x: '' if not isinstance(x, str) else x)

# Now, tokenize the cleaned text data
try:
    tokenized_outputs = tokenizer(list(data_df['description']), padding=True, truncation=True, return_tensors="pt", max_length=512)
    # Proceed with the rest of your code
except Exception as e:
    print(f"An error occurred during tokenization: {e}")

# Extract input_ids and attention_mask
input_ids = tokenized_outputs['input_ids']
attention_mask = tokenized_outputs['attention_mask']

# Extract target labels (assuming 'label_column' is the name of your target label column)
target_labels = torch.tensor(data_df['ratingsCount'].values, dtype=torch.long)

# Create an instance of your dataset
dataset = TensorDataset(input_ids, attention_mask, target_labels)

# Split the dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation
train_loader = create_dataloader(train_dataset, batch_size=32)
val_loader = create_dataloader(val_dataset, batch_size=32, is_train=False)

# Output Dimension for BCELoss with Memory Augmentated Transformer Model for CTR Prediction
class MemoryAugmentedCTRTransformer(nn.Module):
    def __init__(self, base_model, memory_size):
        super().__init__()
        self.base_model = FunnelModel.from_pretrained("funnel-transformer/xlarge-base")
        self.memory = nn.Parameter(torch.randn(memory_size, base_model.config.hidden_size))

    def forward(self, x):
        base_output = self.base_model(x)
        memory_output = base_output + self.memory
        output = self.fc(x)  # No activation here
        return output  # BCEWithLogitsLoss will apply sigmoid
        return base_output

# Training Loop
def train_model(train_loader, val_loader, model, optimizer, loss_fn, epochs, patience, scheduler):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        average_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item()
        average_val_loss = val_loss / len(val_loader)
        
        # Early stopping and learning rate adjustment
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience // 2:
                reduce_learning_rate(optimizer)
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f"Epoch {epoch + 1}: Train Loss = {average_train_loss}, Validation Loss = {average_val_loss}")
        scheduler.step()

    return model, best_val_loss

# Function to reduce learning rate
def reduce_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
    print(f"Reducing learning rate to {param_group['lr']}")

# DataLoader Function
def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Your features and labels
features = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
labels = torch.tensor([0, 1, 0], dtype=torch.float32)

# Create a TensorDataset
dataset = TensorDataset(features, labels, target_labels)

# Splitting the dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Dataset Representation
train_data = torch.randn(1000, 10)  
train_labels = torch.randint(0, 2, (1000,))  
train_dataset = TensorDataset(train_data, train_labels, target_labels)

val_data = torch.randn(200, 10)  
val_labels = torch.randint(0, 2, (200,))  
val_dataset = TensorDataset(val_data, val_labels, target_labels)

# Create DataLoaders for training and validation
train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# Model and Optimizer Setup
model = FunnelModel.from_pretrained("funnel-transformer/xlarge-base")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
patience = 10
epochs = 10

# Function to check performance and adjustment
def check_performance_and_adjust(patience, epoch_number, val_loss, best_val_loss, patience_counter, reduce_learning_rate, optimizer):
    early_stopping_triggered = False
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset patience counter on improvement
        # ... save the best model ...
        # Optional: save checkpoint
    else:
        patience_counter += 1
        if patience_counter == patience // 2:
            reduce_learning_rate(optimizer)
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch_number}") 
            early_stopping_triggered = True

    return best_val_loss, patience_counter, early_stopping_triggered

# Main training Execution
def train_model_with_params(task_data):
    model, train_loader, val_loader, epochs, patience = task_data['model'], task_data['train_loader'], task_data['val_loader'], task_data['epochs'], task_data['patience']
    optimizer = optim.Adam(model.parameters(), lr=task_data['learning_rate'])
    criterion = nn.BCELoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    return train_model(train_loader, val_loader, model, optimizer, criterion, epochs, patience, scheduler)

def main():
    num_processes = os.cpu_count() or 4
    tasks = [
        {'model': model, 'train_loader': train_loader, 'val_loader': val_loader, 'epochs': 10, 'patience': 10, 'learning_rate': 0.001},
        {'model': model, 'train_loader': train_loader, 'val_loader': val_loader, 'epochs': 10, 'patience': 10, 'learning_rate': 0.002}
    ]

    with Pool(processes=num_processes) as pool:
        results = pool.map(train_model_with_params, tasks)
        for result in results:
            # Handle the result
            pass

if __name__ == '__main__':
    cProfile.run('main()')

# Validation Loop
def validate_model(model, val_loader, criterion, device):
    model.to(device)
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, attention_mask, labels in val_loader:
            inputs = torch.tensor(eval(inputs)) if isinstance(inputs, str) else inputs
            attention_mask = torch.tensor(eval(attention_mask)) if isinstance(attention_mask, str) else attention_mask
            labels = torch.tensor(eval(labels)) if isinstance(labels, str) else labels
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(inputs, attention_mask=attention_mask).squeeze()
            if outputs.shape != labels.shape:
                outputs = outputs.view_as(labels)

            loss = criterion(outputs, labels.float())
            total_val_loss += loss.item()

            predictions = (outputs > 0.5).float()  # For binary classification
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Store predictions and labels for calculating AUC and Log Loss
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_samples
    auc_score = roc_auc_score(all_labels, all_predictions)
    logloss = log_loss(all_labels, all_predictions)

    return avg_loss, accuracy, auc_score, logloss

# Model Saving based on validation loss
def save_model_if_improved(model, val_loss, best_val_loss, checkpoint_path):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    return best_val_loss

best_val_loss = save_model_if_improved(model, avg_val_loss, best_val_loss, checkpoint_path)

# Definition of the evaluate_model function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    # Initialization of variables
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in data_loader:
            # Processing and evaluation logic
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    auc_score = roc_auc_score(all_labels, all_predictions)  # Assumes binary classification
    logloss = log_loss(all_labels, all_predictions)

    return avg_loss, accuracy, auc_score, logloss

# Evaluate the Model
avg_val_loss, accuracy, auc_score, logloss = evaluate_model(memory_augmented_funnel, val_loader, criterion, device)

# Print the evaluation metrics
print(f"Average Validation Loss: {avg_val_loss}")
print(f"Accuracy: {accuracy}")
print(f"AUC Score: {auc_score}")
print(f"Log Loss: {logloss}")

# Loading the Amazon Data
data1 = pd.read_csv(r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_data.csv', low_memory=False)
data2 = pd.read_csv(r'C:\Users\ANDLab10\Desktop\EEG\Datasets\Amazon_Books\Books_rating.csv', low_memory=False)

# Merging the datasets on 'Title'
merged_data = pd.merge(data1, data2, on='Title')
merged_data.drop_duplicates(inplace=True)
merged_data.ffill(inplace=True)

# Text Data Concatenation and Tokenization
columns_to_concat = ['description', 'review/summary', 'review/text', 'Title']
merged_data[columns_to_concat] = merged_data[columns_to_concat].fillna('').astype(str)
merged_data['combined_text'] = merged_data[['description', 'review/summary', 'review/text', 'Title']].agg(' '.join, axis=1)

# Tokenization with spaCy
nlp = spacy.load("en_core_web_sm")  # Load the model

def tokenize_with_spacy(text):
    return [token.text for token in nlp(text)]

texts = merged_data['combined_text'].tolist()  # List of texts for tokenization
with Pool(processes=4) as pool:  # Adjust the number of processes as needed
    spacy_results = pool.map(tokenize_with_spacy, texts)

# Tokenization with Funnel Transformer
try:
    tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge-base")
    tokenized_data = tokenizer(merged_data['combined_text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
except Exception as e:
    print(f"Error in tokenization: {e}")# Text Data Concatenation and Tokenization
columns_to_concat = ['description', 'review/summary', 'review/text', 'Title']
merged_data[columns_to_concat] = merged_data[columns_to_concat].fillna('').astype(str)
merged_data['combined_text'] = merged_data[['description', 'review/summary', 'review/text', 'Title']].agg(' '.join, axis=1)

# Tokenization with spaCy
nlp = spacy.load("en_core_web_sm")  # Load the model

def tokenize_with_spacy(text):
    return [token.text for token in nlp(text)]

texts = merged_data['combined_text'].tolist()  # List of texts for tokenization
with Pool(processes=4) as pool:  # Adjust the number of processes as needed
    spacy_results = pool.map(tokenize_with_spacy, texts)

# Tokenization with Funnel Transformer
try:
    tokenizer = FunnelTokenizer.from_pretrained("funnel-transformer/xlarge-base")
    tokenized_data = tokenizer(merged_data['combined_text'].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
except Exception as e:
    print(f"Error in tokenization: {e}")

# Prepare the preprocessed data
def prepare_preprocessed_data(tokenized_data, data, label_column):
    """Prepares preprocessed data for a CTR prediction model.

    Args:
        tokenized_data: A dictionary containing 'input_ids' and 'attention_mask'.
        data: The DataFrame containing the label column.
        label_column: The name of the column containing labels in your data.

    Returns:
        A dictionary containing 'input_ids', 'attention_mask', and 'labels'.

    Raises:
        ValueError: If there is a length mismatch between 'input_ids' and 'attention_mask'.
                    If the label values are not in the expected set (e.g., {0, 1}).
                    If the label column is not found in the DataFrame.
    """
    # Validate input_ids and attention_mask lengths
    if len(tokenized_data['input_ids']) != len(tokenized_data['attention_mask']):
        raise ValueError("Length mismatch: 'input_ids' and 'attention_mask' must be the same length.")

    # Validate label column existence
    if label_column not in data.columns:
        raise ValueError(f"Label column '{label_column}' not found in the data.")

    # Validate label values
    if not set(data[label_column].unique()).issubset({0, 1}):  # Adjust allowed values if needed
        raise ValueError("Invalid labels: Only binary values 0 and 1 are supported.") 

    # Prepare preprocessed data
    preprocessed_data = {
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask'],
        'labels': torch.tensor(data[label_column].values, dtype=torch.long)
    }
    return preprocessed_data

# Define Memory-Augmented Transformer model
class MemoryAugmentedCTRTransformer(nn.Module):
    def __init__(self, base_model, memory_size):
        super().__init__()
        self.base_model = base_model  # This is your base Transformer model
        self.memory = nn.Parameter(torch.randn(memory_size, base_model.config.hidden_size))

    def forward(self, x):
        # Define the forward pass
        x = self.base_model.embeddings(x) + self.memory
        return self.base_model(x)[0]

# Instantiate the model
# Define the funnel_model variable
funnel_model = FunnelBaseModel.from_pretrained("funnel-transformer/xlarge-base")

memory_size = 100  # Specify the memory size
memory_augmented_funnel = MemoryAugmentedCTRTransformer(funnel_model, memory_size)

# Define the configuration for the FunnelModel
funnel_config = FunnelConfig(
    # Define the necessary configuration parameters here
    # Example:
    vocab_size=30522,
    d_model=768,
    n_head=12,
    d_ff=3072,
    block_sizes=[6, 6]
)

# Create an instance of the FunnelModel with this configuration
funnel_model = FunnelModel(config=funnel_config)

# Set up and Move Model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
label_column = 'ratingsCount'
preprocessed_data = {
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask'],
        'labels': torch.tensor(merged_data[label_column].values, dtype=torch.long)
    }
input_ids = torch.tensor(preprocessed_data['input_ids'])
attention_masks = torch.tensor(preprocessed_data['attention_masks'])
labels = torch.tensor(preprocessed_data['labels'])
dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = memory_augmented_funnel.to(device)
loss_function = nn.CrossEntropyLoss()  # Example for classification
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_val_loss = float('inf')

for epoch in range(num_epochs):
    for batch in train_loader:  # Assuming train_loader is your DataLoader
        inputs, labels = batch  # Unpack the batch of data
        inputs, labels = inputs.to(device), labels.to(device) # Move inputs and labels to the same device as the model

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, calculate loss
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

# Dataset
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define evaluating models
def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(probabilities)
            true_labels.extend(labels.cpu().numpy())

    # Return collected predictions and true labels
    return predictions, true_labels

# Evaluate the model
# Assuming val_loader is your validation DataLoader
predictions, true_labels = evaluate_model(model, val_loader, device)

# Compute metrics
auc = roc_auc_score(true_labels, predictions)
loss = log_loss(true_labels, predictions)

print(f"Model AUC: {auc:.4f}, Log Loss: {loss:.4f}")

