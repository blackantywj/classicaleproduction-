'''
Code by: yang
Date: 10/20
'''

import torch
import torch.nn as nn
import math

# 比如说，现在句子长度是5，在后面注意力机制的部分，在计算出来QK转置除以根号之后，softmax之前，我们得到的形状
# len_ipuut × len_input 代表每个单词对其余包含自己的单词的影响力
# 所以这里需要有一个同等大小形状的矩阵，告诉哪些位置是PAD，之后在计算softmax之前会把这里设置为无穷大；
# 注意这里得到的矩阵形状[batch_size × len_q × len_k], 对k中的符号进行标识，并没有对q中的做标识，因为没必要
# seq_q和seq_k不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的；

class MultiHeadAttention(nn.Module):
	def __init__(self):
		super(MultiHeadAttention, self).__init__()
		#输入进来的QKV是相等的，会使用映射Linear做一个映射得到参数矩阵Wq，Wk，Wv
		self.W_Q = nn.Linear(d_model, d_k * n_heads)
		self.W_K = nn.Linear(d_model, d_k * n_heads)
		self.W_V = nn.Linear(d_model, d_v * n_heads)
		
def get_attn_pad_mask(seq_q, seq_k):
	batch_size, len_q = seq_q.size()
	batch_size, len_k = seq_k.size()
	# eq(zero) is PAD token
	pad_attn_mask = sen_k.data.eq(0).unsqueeze(1)
	return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size × len_q × len_k]

class EncoderLayer(nn.Module):
	def __init__(self):
		super(EncoderLayer, self).__init__()
		self.enc_self_attn = MultiHeadAttention()
		self.pos_ffn = PoswiseFeedForwardNet()
	
	def forward(self, enc_inputs, enc_self_attn_mask):
		# 这个是做自注意力层，输入是enc_inouts，形状是[batch_size × seq_len_q × d_model] 注意最初始得QKV矩阵是等同于这个输入得，看一下enc_self_attn函数：6
		enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
		enc_outputs = self.pos_ffn(enc_outputs)
		return enc_outputs, attn

class PositionEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionEncoding, self).__init__()

		#位置编码的实现很简单，直接对照公式敲代码即可，下面的代码只是其中实现的一种方式；
		#需要注意的是偶数和奇数在公式上有一个共同的部分，我们使用log函数把次方拿下来，方便计算；
		#假设dmodel是512，那么公式里的0，1，...，511代表每一个位置，2i符号中i从0取到了255，那么2i对应取值是0，2，4，...，510
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len,d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:,0::2] = torch.sin(position * div_term)  # pe[:,0::2],从0开始到最后，步长为2，代表偶数位置
		pe[:,1::2] = torch.cos(position * div_term)  # 代表奇数位置
		# 上面代码获取之后得到的pe：[max_len * d_model]

		#加一个维度的变换，pe形状变成[max_len * 1 * d_model]
		pe = pe.unsqueeze(0).transpose(0,1)

		self.register_buffer('pe', pe)  # 定义一个缓冲区，简单理解为这个参数不更新

	def forward(self, x):
		"""
		x: [seq_len, batch_size, d_model]
		"""
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model)(output + residual) # [batch_size, seq_len, d_model]

class transformerEncoder(nn.Module):
	def __init__(self, src_vovab_size = , d_model, n_layers):
		super(transformerEncoder, self).__init__()
		self.src_emb = nn.Embedding(src_vocab_size, d_model) # 定义生成一个矩阵，大小是src_vocab_size × d_model
		self.pos_emb = PositionEncoding(d_model) # 位置编码情况，这里是固定的正余弦函数，也可以使用类似的nn.Embedding获得一个可以更新学习的位置编码
		self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来
	
	#实现函数
	def forward(self, enc_inputs):
		# 这里的enc_inputs形状是：[batch_size × source_len]（source_len是编码端输入句子的长度）

		# 下面这个代码是src_emb进行索引定位，enc_outputs输出形状是[batch_size,src_len,d_model]
		enc_outputs = self.src_emb(enc_inputs)

		# 这里就是位置编码，把两者相加放入到了这个函数里面，从这里可以去看一下位置编码函数的实现：3
		enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)

		# get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，这个函数的实现：4
		enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
		enc_self_attns = []
		for layer in self.layers:
			# 看EncoderLayer函数：5
			enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
			enc_self_attns.append(enc_self_attn)
		return enc_outputs, enc_self_attns
class PGN(nn.Module):
    def __init__(
        self,
		
        ):
        super(self, PGN).__init__()
	    self.encoder = transformerEncoder()
    def forward(self, region_feature, structure_variable, content_variable):
        pass

# #句子的输入部分
# sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

# #transformer parameters
# #padding should be zero
# #构建词表
# #编码器词表
# src_vocab = {'P':0,'ich':1,'mochte':2,'ein':3,'bier':4}
# src_vocab_size = len(src_vocab)

# #解码器词表
# tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4}
# tgt_vocab_size = len(tgt_vocab)

# src_vocab = 5 #length of source
# tgt_vocab = 5 #length of target

# #模型参数
# d_model = 512  #每个词转成embedding时的大小
# d_ff = 2048  #前馈神经网络中映射到多少维度
# d_k = d_v = 64 #
# n_layers = 6 # number of encoder of decoder layer
# n_heads = 8  #多头注意力机制的头的数量

# tf = transformerEncoder()
# print(tf)