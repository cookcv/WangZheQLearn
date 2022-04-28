import torch
#from torchtext import data
import numpy as np


# patch on Torchtext's batching process that makes it more efficient
# from http://nlp.seas.harvard.edu/2018/04/03/attention.html#position-wise-feed-forward-networks

# class MyIterator(data.Iterator):
#     def create_batches(self):
#         if self.train:
#             def pool(d, random_shuffler):
#                 for p in data.batch(d, self.batch_size * 100):
#                     p_batch = data.batch(
#                         sorted(p, key=self.sort_key),
#                         self.batch_size, self.batch_size_fn)
#                     for b in random_shuffler(list(p_batch)):
#                         yield b
#             self.batches = pool(self.data(), self.random_shuffler)
#
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size,
#                                           self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))

