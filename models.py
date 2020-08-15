import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor

else:
    longTensor = torch.LongTensor
    floatTensor = torch.FloatTensor


class TransEModel(nn.Module):
	def __init__(self, config):
		super(TransEModel, self).__init__()
		self.learning_rate = config.learning_rate
		self.early_stopping_round = config.early_stopping_round
		self.L1_flag = config.L1_flag
		self.filter = config.filter
		self.embedding_size = config.embedding_size
		self.entity_total = config.entity_total
		self.relation_total = config.relation_total
		self.batch_size = config.batch_size

		self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
		self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
		nn.init.xavier_uniform_(self.ent_embeddings.weight)
		nn.init.xavier_uniform_(self.rel_embeddings.weight)

		normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
		normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
		self.ent_embeddings.weight.data = normalize_entity_emb
		self.rel_embeddings.weight.data = normalize_relation_emb

	def forward(self, pos_h, pos_t, pos_r, neg_h, neg_t, neg_r):
		"""
		link predicitons
		"""
		pos_h_e = self.ent_embeddings(pos_h)
		pos_t_e = self.ent_embeddings(pos_t)
		pos_r_e = self.rel_embeddings(pos_r)
		neg_h_e = self.ent_embeddings(neg_h)
		neg_t_e = self.ent_embeddings(neg_t)
		neg_r_e = self.rel_embeddings(neg_r)

		# L1 distance
		if self.L1_flag:
			pos = torch.sum(torch.abs(pos_h_e + pos_r_e - pos_t_e), 1)
			neg = torch.sum(torch.abs(neg_h_e + neg_r_e - neg_t_e), 1)
		# L2 distance
		else:
			pos = torch.sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1)
			neg = torch.sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1)
		return pos, neg


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, pos, neg, margin):
        zero_tensor = floatTensor(pos.size())
        zero_tensor.zero_()
        zero_tensor = autograd.Variable(zero_tensor)
        return torch.sum(torch.max(pos - neg + margin, zero_tensor))

def orthogonalLoss(rel_embeddings, norm_embeddings):
    return torch.sum(torch.sum(norm_embeddings * rel_embeddings, dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def NormLoss(embeddings, dim=1):
    norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
    return torch.sum(torch.max(norm - autograd.Variable(floatTensor([1.0])), autograd.Variable(floatTensor([0.0]))))
    