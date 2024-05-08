import torch
def precompute_rotate_z(seq_len, dim, base_value=10000):
    """
            1. 构造基于位置的复数向量，如何构造呢？
            1. 维度1～d/2 构造freq频率
            2. 位置pos 1～n 和 freq做外积（笛卡尔积）求得旋转角度 theta
            3. 用位置和构造theta角度向量
            3. 利用theta角度构造复数向量 z = cos(theta) + i*sin(theta)
    """
    # 1. 维度1～d/2 构造freq频率
    # freq = 1/10000**2i/dim ,i = (0,2,4,...,dim/2)
    # 从dim大小向量 压缩到 dim/2大小的向量

    dim_indexes = torch.arange(0, dim, 2)  # 0，2，4，...,dim
    dim_indexes = dim_indexes[:dim // 2]  # 对最后元素做处理
    t = dim_indexes / dim  # 转换到0~1
    freq = 1 * base_value ** (-t)

    # 2. 位置pos 1～n 和 freq做外积（笛卡尔积）求得旋转角度 theta
    pos = torch.arange(start=0, end=seq_len)
    theta = torch.outer(pos, freq)

    # 3. 用theta角度构造复数向量 z = cos(theta) + i*sin(theta), 用于旋转的复数z矩阵
    rotate_z = torch.polar(torch.ones_like(theta), theta)
    return rotate_z


def compute_sin_pos_emb(seq_len, dim, base_value):
    rotate_z = precompute_rotate_z(seq_len, dim, base_value)
    sin_emb = torch.view_as_real(rotate_z)
    sin_emb = sin_emb.reshape((seq_len, dim))
    return sin_emb


# class Rope(nn.Module): # 没引入任何需要学习的参数，没有必要继承nn.module
class Rope(object):
    def __init__(self, base_value=10000):
        super().__init__()
        self.base_value = base_value

    def apply_rotatory_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入q，
        返回 旋转后的q
        1. 构造基于位置的复数向量，如何构造呢？
            1. 维度1～d/2 构造freq频率
            2. 位置pos 1～n 和 freq做外积（笛卡尔积）求得旋转角度 theta
            3. 用位置和构造theta角度向量
            3. 利用theta角度构造复数向量 z = cos(theta) + i*sin(theta)
        2. 把q变换到复数空间
            1. 相邻两两一对转换成复数，q0,q1 -> q0 + i*q1=z1, (q0,q1,q2,...q_d) -> (z1,z2,...,z_d/2)
        3. 复数空间相乘，做旋转(单位复数相乘就是对复数做旋转)
        4. 再变换回实数空间 向量 (实部，虚部)
        """

        dim = x.size(-1)
        seq_len = x.size(-2)  # n is seq_len
        rotate_z = precompute_rotate_z(seq_len, dim, self.base_value)

        # 把q变换到复数空间
        # xq = input_ids.float().reshape(*input_ids.shape[:-1], -1, 2)
        xq = x.float().view(*x.shape[:-1], -1, 2)
        z = torch.view_as_complex(xq)
        """
                xq = input_ids.float().view(*input_ids.shape[:-1], -1, 2)
                这个是把最后一个维度，变成两个维度，确保最后一个维度是2，成对出现
                q0,q1,q2,q3,...,qn-1,qn -> 
                [[q0,q1],        -> z0
                 [q2,q3],        -> z1       
                 [q4,q5],        -> z2
                 ...
                 [qn-1,qn]]      -> z_n//2         
        """

        # 复数空间相乘，做旋转(单位复数相乘就是对复数做旋转)
        rotated_z = z * rotate_z

        # 再变换回实数空间 向量 (实部，虚部)
        rotated_x = torch.view_as_real(rotated_z)
        # 在实数空间中扁平化张量，确保从第 2 维开始扁平化
        rotated_x = rotated_x.flatten(start_dim=-2)
        """
        view_as_real
        这个函数会把
        z=re_z + im_z*j
        re_z0+im_z0 * j -> [re_z0,im_z0]
        re_z1+im_z1 * j -> [re_z1,im_z1]
        ...             
        re_z_n//2 +im_z_n//2 * j -> [re_z_n//2,im_z_n//2]

        flatten(-2)会把最后两个维度展平
        [re_z0,im_z0,re_z1,im_z1,...,re_z_n//2,im_z_n//2]  大小又便会n，跟输入x的最后一个维度大小一样ndim

        """
        return rotated_x