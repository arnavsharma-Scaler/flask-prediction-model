import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
from sklearn.mixture import GaussianMixture


class EmbeddingClassi():
  def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", PATH = '', datapath='Customer.csv'):
    self.path = PATH
    self.df = pd.read_csv(PATH+datapath).dropna()
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.labels = self.Pretrain_model()

  def query(self, q):
    encoded_input = self.tokenizer(q, padding=True, truncation=True, return_tensors='pt')
    encoded_input.to('cpu')
    # Compute token embeddings
    with torch.no_grad():
      model_output = self.model(**encoded_input)
    # Perform pooling
    sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return self.pca.transform(sentence_embeddings.cpu().detach().numpy())

  def mean_pooling(self,model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  def Pretrain_model(self,model_name="sentence-transformers/all-MiniLM-L6-v2"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)
    self.PCA_compute()
    x = self.Clustering()
    print(x)
    return x

  def PCA_compute(self):
    self.embedding = np.load(self.path+'dummy_embeddings.npy')
    pca = PCA()
    pca.fit(self.embedding)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum>=0.95) + 1
    self.pca = PCA(n_components = d)
    self.pca_embeddings = self.pca.fit_transform(self.embedding)

  def Clustering(self,n_components=5):
    gmm = GaussianMixture(n_components=n_components).fit(self.pca_embeddings)
    self.labels = gmm.predict(self.pca_embeddings)
    return self.labels