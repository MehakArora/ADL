import pandas as pd
import re
from unicodedata import normalize
import torch
from torchnlp.encoders.text import TreebankEncoder
from torchnlp.encoders.text import pad_tensor, stack_and_pad_tensors
from torchnlp.encoders.text import StaticTokenizerEncoder

def french_tokenize(text):
    text = text.replace('!', ' ').replace('"', ' ').replace('#', ' ') \
            .replace('$', ' ').replace('%', ' ').replace('&', ' ') \
            .replace('(', ' ').replace(')', ' ').replace('*', ' ') \
            .replace('+', ' ').replace(',', ' ').replace('-', ' ') \
            .replace('.', ' ').replace('/', ' ').replace(':', ' ') \
            .replace(';', ' ').replace('<', ' ').replace('=', ' ') \
            .replace('>', ' ').replace('?', ' ').replace('@', ' ') \
            .replace('\\', ' ').replace('^', ' ').replace('_', ' ') \
            .replace('`', ' ').replace('{', ' ').replace('|', ' ') \
            .replace('}', ' ').replace('~', ' ').replace('\t', ' ') \
            .replace('\n', ' ')
    return text.split()


def custom_pad_sequences(sequences, max_len, padding_value=0):
    padded_seqs = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad the sequence
            padded = torch.cat([seq, torch.tensor([padding_value] * (max_len - len(seq)), dtype=torch.long)])
        else:
            # Truncate if longer than max_len
            padded = seq[:max_len]
        padded_seqs.append(padded)
    return torch.stack(padded_seqs)

def decode_sequence(sequence, tokenizer):
    # Convert tensor to list if necessary
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    
    # Decode the sequence using the tokenizer's vocabulary
    decoded_sequence = [tokenizer.index_to_token[token] for token in sequence if token in tokenizer.index_to_token]
    
    # Join the tokens to form the final sentence
    return ' '.join(decoded_sequence)

def clean_text(text):
    text = normalize('NFD', text.lower())
    text = re.sub('[^A-Za-z ]+', '', text)
    return text

def clean_and_prepare_text(text):
    text = '[start] ' + clean_text(text) + ' [end]'
    return text


df = pd.read_csv('Data/en-fr.txt', names=['en', 'fr', 'attr'], usecols=['en', 'fr'], sep='\t')
df = df.sample(frac=1, random_state=42)
df = df.reset_index(drop=True)
df.head()



df['en'] = df['en'].apply(lambda row: clean_text(row))
df['fr'] = df['fr'].apply(lambda row: clean_and_prepare_text(row))


en = df['en']
fr = df['fr']

en_max_len = max(len(line.split()) for line in en)
fr_max_len = max(len(line.split()) for line in fr)
sequence_len = max(en_max_len, fr_max_len)

print(f'Max phrase length (English): {en_max_len}')
print(f'Max phrase length (French): {fr_max_len}')
print(f'Sequence length: {sequence_len}')

entokenizer = TreebankEncoder(en)
frtokenizer = StaticTokenizerEncoder(fr, tokenize=french_tokenize, append_eos=False, reserved_tokens=['<pad>'])

en_sequences = [torch.tensor(entokenizer.encode(sentence)) for sentence in en]
fr_sequences = [torch.tensor(frtokenizer.encode(sentence)) for sentence in fr]

# Pad the sequences to the desired length
en_x = custom_pad_sequences(en_sequences, sequence_len, padding_value=0)
fr_y = custom_pad_sequences(fr_sequences, sequence_len + 1, padding_value=0)

en_vocab_size = len(entokenizer.vocab) + 1
fr_vocab_size = len(frtokenizer.vocab) + 1

print(f'Vocabulary size (English): {en_vocab_size}')
print(f'Vocabulary size (French): {fr_vocab_size}')

inputs = { 'encoder_input': en_x, 'decoder_input': fr_y[:, :-1] }
outputs = fr_y[:, 1:]

decoded_en_sentence = decode_sequence(en_x[0], entokenizer)
decoded_fr_sentence = decode_sequence(fr_y[0], frtokenizer)

print(f'Decoded English sentence: {decoded_en_sentence}')
print(f'Decoded French sentence: {decoded_fr_sentence}')
