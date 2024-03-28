# from typing import Any
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# class BilingualDataset(Dataset):
#     def __init__(self, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, ds, seq_len)->None:
#         self.ds = ds
#         self.tokenizer_src = tokenizer_src
#         self.tokenizer_tgt = tokenizer_tgt
#         self.seq_len = seq_len
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang
        
#         self.sos_token = torch.Tensor([tokenizer_src.token_to_id(["[SOS]"])], dtype = torch.int64)
#         self.eos_token = torch.Tensor([tokenizer_src.token_to_id(["[EOS]"])], dtype = torch.int64)
#         self.pad_token = torch.Tensor([tokenizer_src.token_to_id(["[PAD]"])], dtype = torch.int64)
       

    
    
#     def __len__(self):
#         return len(self.ds)
    
    
    
#     def __getitem__(self, index: Any) -> Any:
#         src_target = self.ds[index]
#         src_text = src_target["translation"][self.src_lang]
#         tgt_text = src_target["translation"][self.tgt_lang]
        
#         enc_input_tokens = self.tokenizer_src.encode(src_text).ids
#         dec_intput_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
#         #padding bhi karlo lage hath
#         enc_padding = self.seq_len - len(enc_input_tokens) - 2 #start and end token
#         dec_padding = self.seq_len - len(dec_intput_tokens) - 1 #end token
        
#         if enc_padding < 0 or dec_padding < 0:
#             raise ValueError("lamba hai ni medam lambaa") #sahi kardena bhai majak nahi 
        
#         #start token, end token, pad token
#         encoder_input = torch.cat([self.sos_token, 
#                                    torch.Tensor(enc_input_tokens), 
#                                    self.eos_token, 
#                                    torch.Tensor([self.pad_token]*enc_padding, dtype = torch.int64)]
#         )
#         # start token pad token
#         decoder_input = torch.cat([self.sos_token, 
#                                    torch.Tensor(dec_intput_tokens), 
#                                    torch.Tensor([self.pad_token]*dec_padding, dtype = torch.int64)]
#         )
#         #end (expected output)
#         label = torch.cat([torch.Tensor(dec_intput_tokens), 
#                            self.eos_token, 
#                            torch.Tensor([self.pad_token]*dec_padding, dtype = torch.int64)]
#         )
        
        
#         # assert encoder_input.shape == decoder_input.shape
#         assert encoder_input.size(0) == self.seq_len
#         assert decoder_input.size(0) == self.seq_len
#         assert label.size(0) == self.seq_len   
        
#         return {
#             "encoder_input": encoder_input,#seq_len
#             "decoder_input": decoder_input,#seq_len
#             "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1, 1, seq_len)
#             "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))  #(1, 1, seq_len)  }
#         }
        
# def causal_mask(size):
#     mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
#     return mask == 0
        
        
        
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)










    def __len__(self):
        return len(self.ds)







    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Lamba hai ni meedam lamba!")

        # Add <s> and </s> token
        encoder_input = torch.cat([self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)],dim=0,)

        # Add only <s> token
        decoder_input = torch.cat([self.sos_token,torch.tensor(dec_input_tokens, dtype=torch.int64),
                                   torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],dim=0,)

        # Add only </s> token
        label = torch.cat([torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],dim=0,)

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
        
        
        
        
        
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
        