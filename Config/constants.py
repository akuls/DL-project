import torch
HAVE_CUDA = torch.cuda.is_available()
BUCKET_SIZE = 256
EMBEDDING_DIM = 300
SIDELENGTH = 50