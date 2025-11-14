import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.data = self.process_data(data_folder, split, self.tokenizer)
    
    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)
        
        # Load SQL queries (only for train and dev, not test)
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)
        else:
            sql_lines = [None] * len(nl_lines)
        
        return list(zip(nl_lines, sql_lines))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nl_query, sql_query = self.data[idx]
        
        # Tokenize encoder input (natural language)
        encoder_inputs = self.tokenizer(
            nl_query,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        
        if self.split == 'test':
            # For test set, we don't have SQL queries
            return {
                'encoder_input_ids': encoder_inputs['input_ids'],
                'encoder_attention_mask': encoder_inputs['attention_mask'],
            }
        else:
            # Tokenize decoder input and target (SQL query)
            decoder_inputs = self.tokenizer(
                sql_query,
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors=None
            )
            
            sql_token_ids = decoder_inputs['input_ids']
            
            # Decoder input: prepend pad_token_id (BOS) to create right-shifted sequence for teacher forcing
            # If SQL is [A, B, C], decoder_input should be [pad, A, B, C]
            decoder_input_ids = [self.tokenizer.pad_token_id] + sql_token_ids
            
            # Decoder target: the SQL query tokens + EOS (what we want to predict at each position)
            # Length should match decoder_input_ids: [A, B, C, EOS]
            decoder_targets = sql_token_ids + [self.tokenizer.eos_token_id]
            
            # Ensure lengths match (they should now both be len(sql_token_ids) + 1)
            
            # Initial decoder input is just the BOS token (pad_token_id)
            initial_decoder_input = self.tokenizer.pad_token_id
            
            return {
                'encoder_input_ids': encoder_inputs['input_ids'],
                'encoder_attention_mask': encoder_inputs['attention_mask'],
                'decoder_input_ids': decoder_input_ids,
                'decoder_targets': decoder_targets,
                'initial_decoder_input': initial_decoder_input,
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_input_ids = [torch.tensor(item['encoder_input_ids']) for item in batch]
    encoder_attention_mask = [torch.tensor(item['encoder_attention_mask']) for item in batch]
    decoder_input_ids = [torch.tensor(item['decoder_input_ids']) for item in batch]
    decoder_targets = [torch.tensor(item['decoder_targets']) for item in batch]
    initial_decoder_inputs = [item['initial_decoder_input'] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_targets_padded = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.tensor(initial_decoder_inputs)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets_padded, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_input_ids = [torch.tensor(item['encoder_input_ids']) for item in batch]
    encoder_attention_mask = [torch.tensor(item['encoder_attention_mask']) for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)
    
    # For test set, initial decoder input is pad_token_id (which is 0)
    initial_decoder_inputs = torch.full((len(batch),), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x