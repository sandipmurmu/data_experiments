import torch
import torch.nn as nn
import math
import torch.nn.functional as F







sentence = "the quick brown fox jumps over a lazy dog"
# convert the words into integers
s_dict = {s: i for i, s in enumerate(sorted(sentence.replace(',', '').split()))}
#print(s_dict)

# convert into integers into tensor
arr_nums = [s_dict[i] for i in sentence.replace(',','').split()]
s_tensor = torch.tensor(arr_nums)
#print(s_tensor)

#create a vocab
vocab_size = 50000
torch.manual_seed(123)
embedder = nn.Embedding(vocab_size,3)
embedded_sentence = embedder(s_tensor).detach()
#print(embedded_sentence)
#print(embedded_sentence.shape)
# dimension of the tensor is number of columns
d = embedded_sentence.shape[1]
# query, key, value
d_q, q_k, q_v = 2, 2, 4
w_query = torch.nn.Parameter(torch.rand(d, d_q))
w_key = torch.nn.Parameter(torch.rand(d, q_k))
w_value = torch.nn.Parameter(torch.rand(d,q_v))

query = embedded_sentence @ w_query
key = embedded_sentence @ w_key
value = embedded_sentence @ w_value



class SelfAttention(nn.Module):

    def __init__(self, d, d_q, d_k, d_v):
        super(SelfAttention,self).__init__()
        self.d = d
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.w_query = torch.nn.Parameter(torch.rand(d,d_q))
        self.w_key = torch.nn.Parameter(torch.rand(d,d_q))
        self.w_value = torch.nn.Parameter(torch.rand(d, d_v))


    def forward(self, x):
        Q = x @ self.w_query
        K = x @ self.w_key
        V = x @ self.w_value
        attention_scores = Q @ K.T / math.sqrt(self.d_k)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = attention_weights @ V
        return context_vector

sa = SelfAttention(d=3, d_q=2, d_k=2, d_v=4)
cv = sa(embedded_sentence)
print(cv.shape)
print(cv)