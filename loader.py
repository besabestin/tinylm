import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(1345)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = {}
    
    def create_vocab(self, full_text):
        self.vocab = set(full_text)
        self.word2idx = {word: idx for idx, word in enumerate(sorted(list(self.vocab)))}
        self.idx2word = {self.word2idx[word]: word for word in self.vocab}

    def create_vocab_most_common(self, full_text, freq_prop):
        frequency = {}
        highest_frequency = 0
        total_words = 0
        dict_words = []
        for word in full_text:
            if word not in frequency:
                frequency[word] = 0
                total_words += 1
            frequency[word] += 1
            highest_frequency = max(highest_frequency, frequency[word])

        print(f'total words: {total_words}')
        frequency_to_word = {i:[] for i in range(1, highest_frequency + 1)}
        for word in frequency:
            frequency_to_word[frequency[word]].append(word)
        print(f'most frequent few words: {frequency_to_word[highest_frequency]} {frequency_to_word[highest_frequency - 1]} {frequency_to_word[highest_frequency - 2]}')
        
        for i in range(highest_frequency, 0, -1):
            for word in frequency_to_word[i]:
                dict_words.append(word)
                if (len(dict_words) >= int(freq_prop * total_words)):
                    break
            
            if (len(dict_words) >= int(freq_prop * total_words)):
                break

        print('done mapping')
        
        # for i in range(highest_frequency, 0, -1):
        #     for word in frequency:
        #         if frequency[word] == i:
        #             dict_words.append(word)
        #             if (len(dict_words) >= int(freq_prop * total_words)):
        #                 break
            
        #     if (len(dict_words) >= int(freq_prop * total_words)):
        #         break

        self.word2idx = {word: idx for idx, word in enumerate(sorted(dict_words))}
        self.idx2word = {self.word2idx[word]: word for word in dict_words}
        self.vocab = set(dict_words)
    
    def all_in_dict(self, words):
        return set(words).issubset(self.vocab)

    def encode(self, words):
        return [self.word2idx[word] for word in words]

    def decode(self, indices):
        return [self.idx2word[idx] for idx in indices]

    def __len__(self):
        return len(self.vocab)


# batch_size,
# context_size,
class Corpus(object):
    def __init__(self, paths, batch_size=32, context_size=32) -> None:
        self.dictionary = Dictionary()
        self.words = []
        self.batch_size = batch_size
        self.context_size = context_size
        self.train_data = {}
        self.val_data = {}
        self.tokenize(paths)
        # self.prepare(self.words)
        self.prepare_most_common(self.words)


    def tokenize(self, paths):
        # Add words to the dictionary
        for path in paths:
            assert os.path.exists(path)
            with open(path, 'r') as f:
                for line in f:
                    words = re.findall(r"[\w']+|[.,!?;]", line) + ['<eol>']
                    # words = line.split() + ['<eol>']
                    # self.words.extend([word.lower() for word in words])
                    self.words.extend(words)
        # self.dictionary.create_vocab(self.words)
        self.dictionary.create_vocab_most_common(self.words, 0.04)
        print(f"vocab size: {len(self.dictionary)}")


    def prepare(self, words):
        print(f'inspect words: {len(words)} {words[0:8]}')
        #OMIT: following two lines
        inputs = torch.stack([torch.tensor(self.dictionary.encode(words[i:i+self.context_size]), dtype=torch.long) for i in range(len(words) - self.context_size - 1)])
        outputs = torch.stack([torch.tensor(self.dictionary.encode(words[i:i+self.context_size]), dtype=torch.long) for i in range(1,len(words) - self.context_size)])
        #randomize the order of the datasets
        _perm = torch.randperm(inputs.shape[0])
        train_split = int(0.8*inputs.shape[0])

        self.train_data = {
            'X': inputs[_perm][:train_split, :],
            'y': outputs[_perm][:train_split, :]
        }
        self.val_data = {
            'X': inputs[_perm][train_split:, :],
            'y': outputs[_perm][train_split:, :]
        }

    def prepare_most_common(self, words):
        # consider only those 
        input_list = []; output_list = []
        for i in range(len(words) - self.context_size - 1):
            if self.dictionary.all_in_dict(words[i:i+self.context_size+1]):
                input_list.append(torch.tensor(self.dictionary.encode(words[i:i+self.context_size]), dtype=torch.long))
                output_list.append(torch.tensor(self.dictionary.encode(words[(i+1):(i+self.context_size+1)]), dtype=torch.long))
        inputs = torch.stack(input_list)
        outputs = torch.stack(output_list)    
        _perm = torch.randperm(inputs.shape[0])
        train_split = int(0.8*inputs.shape[0])

        print(f'total input length: {len(inputs)}')

        self.train_data = {
            'X': inputs[_perm][:train_split, :],
            'y': outputs[_perm][:train_split, :]
        }
        self.val_data = {
            'X': inputs[_perm][train_split:, :],
            'y': outputs[_perm][train_split:, :]
        }
        
    #OMIT: whole get_batch function
    def get_batch(self, _stage):
        dataset = self.train_data if _stage == 'train' else self.val_data
        _perm = torch.randperm(dataset['X'].shape[0])
        return dataset['X'][_perm][:self.batch_size, :].to(device), dataset['y'][_perm][:self.batch_size, :].to(device)