import torch
class CacheAwasteMemV1(object):
    def __init__(self,max_batch_size:int,max_seq_len:int,n_head:int,head_dim:int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_dim = head_dim
        self.cache = torch.zeros(  # 注意力中的缓存
                max_batch_size,
                max_seq_len,
                n_head,
                head_dim,
            )
        self.start_pos = 0

    # 直接实现一个cache类，这个cache类能够支持,save_to_cache，get两个方法
    def save_to_cache(self,new_tensor: torch.Tensor) -> None:
        batch_size,seq_len,n_head,head_dim = new_tensor.shape
        self.cache[:batch_size,self.start_pos:self.start_pos+seq_len] = new_tensor
        self.start_pos = self.start_pos + seq_len

    def get_all_from_cache(self) -> torch.Tensor:
        return self.cache[:,:self.start_pos,:,:]

    # 实现cachekv，要记住就是主要针对inference场景就可以，每次都实现单独一个token的解码
    # 1. 把token的kv存到cachekv中
    # 2. 把从cachekv中把历史的kv和最新的kv一起取出来

class CacheHighFrequecyAllocateMem(object):
    """
    这里还是有个问题，不断分配内存的问题，self.cache = torch.cat((self.cache
    这个操作过于频繁，会导致不断地分配内存，这会导致低效的内存管理
    我想应该取得一个平衡，每次分配一个固定大小的内存，而不是每次来一个tensor就分配一次内存
    比如，每次长度不够时，分配一个固定长度的内存，而不是每次都分配一个token长度的内存
    针对llm inferece的实际应用场景来说，每次都分配5个就可以了，这样就避免了不断反复的分配内存的问题

    """
    def __init__(self, max_batch_size: int, max_seq_len: int, n_head: int, head_dim: int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_dim = head_dim
        self.cache = torch.zeros(max_batch_size, 0, self.n_head, self.head_dim)  # 初始化为空张量
        self.start_pos = 0

    def save_to_cache(self, new_tensor: torch.Tensor) -> None:
        batch_size, seq_len, n_head, head_dim = new_tensor.shape
        # 如果缓存不够大，增加新的空间
        if self.start_pos + seq_len > self.cache.size(1):
            self.cache = torch.cat((self.cache, torch.zeros(self.max_batch_size, seq_len, self.n_head, self.head_dim)), dim=1)
        # 将新的张量拼接到缓存上
        self.cache[:batch_size, self.start_pos:self.start_pos+seq_len] = new_tensor
        self.start_pos += seq_len

    def get_all_from_cache(self) -> torch.Tensor:
        return self.cache[:, :self.start_pos]

class CacheLowFrequecyAllocateMem(object):
    """
    这里还是有个问题，不断分配内存的问题，self.cache = torch.cat((self.cache
    这个操作过于频繁，会导致不断地分配内存，这会导致低效的内存管理
    我想应该取得一个平衡，每次分配一个固定大小的内存，而不是每次来一个tensor就分配一次内存
    比如，每次长度不够时，分配一个固定长度的内存，而不是每次都分配一个token长度的内存
    针对llm inferece的实际应用场景来说，每次都分配5个就可以了，这样就避免了不断反复的分配内存的问题
    为了实现这个低频次分配内存的版本，需要设定一个固定大小的长度
    seq_len_increment_for_cache_allocate_by_default
    每次增加这么多的空间就好了
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, n_head: int, head_dim: int,seq_len_increment_for_cache_allocate=5):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_head = n_head
        self.head_dim = head_dim
        self.cache = torch.zeros(max_batch_size, 0, self.n_head, self.head_dim)  # 初始化为空张量
        self.start_pos = 0
        self.seq_len_increment_for_cache_allocate_by_default = seq_len_increment_for_cache_allocate

    def save_to_cache(self, new_tensor: torch.Tensor) -> None:

        batch_size, seq_len, n_head, head_dim = new_tensor.shape
        assert batch_size<=self.max_batch_size,'数据x的batch size必须<=缓存本身的max_batch_size！！'
        # 如果缓存不够大，增加新的空间
        if self.start_pos + seq_len > self.cache.size(1):
            # 要做个判断来选择要增加的长度，确保一次增加的缓存满足输入new_tensor的seq_len的长度要求
            if self.seq_len_increment_for_cache_allocate_by_default < seq_len:
                cache_increment_len = seq_len
            else:
                cache_increment_len = self.seq_len_increment_for_cache_allocate_by_default

            increment_cache = torch.zeros(self.max_batch_size,
                                          cache_increment_len,
                                          self.n_head, self.head_dim)
            self.cache = torch.cat((self.cache,increment_cache),dim=1)
        # 将新的张量拼接到缓存上
        self.cache[:batch_size, self.start_pos:self.start_pos + seq_len] = new_tensor
        self.start_pos += seq_len
        self.batch_size = batch_size
        # print(f'cache size is {self.cache.shape}')
        """
         其实针对llm场景，根本无需使用max_batch_size,直接使用数据的batch_size就可以了
         按照推理数据的batch_size来扩展就可以了 一开始初始化时，又不知道多大
         算了，作为一个todo吧，日后再进行修改，现在不需要做什么了！
        """

    def get_all_from_cache(self) -> torch.Tensor:
        assert self.batch_size > 0,'如果有效存如数据后，batch_size就会是存入数据x的batch_size'
        return self.cache[:self.batch_size, :self.start_pos]