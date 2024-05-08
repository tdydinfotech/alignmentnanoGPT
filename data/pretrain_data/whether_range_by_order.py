import numpy as np
token_ids = [4145, 2153, 1049, 6203, 6827, 2883, 4819, 2350, 1906, 1552, 2404, 2159,8,    0,  138, 4790, 5251, 4008,  117, 1239, 1239, 6827, 2703, 4813,1321, 1425, 4630,  885,  890,    8,    0, 1037, 2946, 2213, 1038, 6240,774, 6528, 6827, 1359, 2593, 3626, 1296, 4645, 1896, 1866,    8,    0,1820, 6129,  867, 6387, 4809, 1804, 5538, 6827, 5877, 5888, 1261,   51,1670,  380, 4352,    8]
# data_dir = './'
# data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='rank')
import pickle
meta_path = './'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
# TODO: 使这更通用以适用于任意的编码器/解码器方案
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(decode(token_ids))
