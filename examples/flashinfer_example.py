
import torch

import flashinfer

nnz_kv = 10

num_kv_heads = 1

head_dim = 128

# the key, value, tensor need to be added into the cache
k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)

v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)

# 45 + 8 + 25 + 22 = nnz_kv

# a list of request token length [3, 7] => two req with len 3 and len 7 
kv_append_length = torch.tensor([3,7], dtype=torch.int32, device="cuda:0")

# = [0, len_req1, len_req1 + len_req2, ...]
kv_append_indptr = torch.cat(

    [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]

).int()  # [0, 45, 53, 78, 100]

print(kv_append_indptr)

max_num_pages = 3

page_size = 4

# this is the actual memory table that used to save the key value tensor
paged_kv_cache = torch.zeros(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0)

# number of pages needed to save the key value tensor for each request
num_pages_per_req = torch.tensor([1, 2], dtype=torch.int32, device="cuda:0")

# page indices pointer [0, 1, 3] => first req use 1 page (page 0), snd request use 2 pages(paeg 1 & 2)
kv_page_indptr = torch.cat(

    [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]

).int()

print(kv_page_indptr)

#  use first 3 pages in the paged-kv

kv_page_indices = torch.arange(3, dtype=torch.int32, device="cuda:0")
print(kv_page_indices)
# 45 = (3 - 1) * 16 + 13

# 8 = (1 - 1) * 16 + 8

# 25 = (2 - 1) * 16 + 9

# 22 = (2 - 1) * 16 + 6

# the number of slot being used by each request in each request's last page
kv_last_page_len = torch.tensor([3, 3], dtype=torch.int32, device="cuda:0")

seq_len= flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size)
print(seq_len)

batch_indices, positions = flashinfer.get_batch_indices_positions(

    kv_append_indptr, seq_len, nnz_kv

)
print(batch_indices)
print(positions)

flashinfer.append_paged_kv_cache(

    k_append,

    v_append,

    batch_indices,

    positions,

    paged_kv_cache,

    kv_page_indices,

    kv_page_indptr,

    kv_last_page_len

)

print(paged_kv_cache)

print(paged_kv_cache[kv_page_indices]) # [num_page, 2, page_size, num_head, head_dim]

k_cache = paged_kv_cache[:,0,:,:,:]
v_cache = paged_kv_cache[:,1,:,:,:]

print(k_cache.shape)
extracted_k = k_cache[0:1,0:3,:,:]
print(extracted_k.squeeze(0).shape)
print(k_append[0:3].shape)
assert torch.equal(extracted_k.squeeze(0), k_append[0:3])

print(k_cache.shape)
print(k_cache[1:2,:,:,:].shape, k_cache[2:3,0:3,:,:].shape)
extracted_k = torch.cat([k_cache[1:2,:,:,:],k_cache[2:3,0:3,:,:]],dim=1)
print(extracted_k.squeeze(0).shape)
print(k_append[3:10].shape)
assert torch.equal(extracted_k.squeeze(0), k_append[3:10])



