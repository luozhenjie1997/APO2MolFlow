import torch
import scipy
import numpy as np
import enum
from model.utils.scoring import *


# 计算向量之间的角度
def th_ang_v(ab, bc, eps: float = 1e-8):
    def th_norm(x, eps: float = 1e-8):  # 计算向量的欧几里得长度
        return x.square().sum(-1, keepdim=True).add(eps).sqrt()

    def th_N(x, alpha: float = 0):  # 向量归一化。将向量方向保留，长度化为1
        return x / th_norm(x).add(alpha)

    ab, bc = th_N(ab), th_N(bc)  # 归一化向量
    # 计算角度分量
    cos_angle = torch.clamp((ab * bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1 - cos_angle.square() + eps)
    """不直接输出角度而输出分量，可以避免因为角度限定在一个周期内导致的的不连续性，同时唯一确定正负角"""
    dih = torch.stack((cos_angle, sin_angle), -1)
    return dih

# 计算向量间的二面角，
def th_dih_v(ab, bc, cd):
    def th_cross(a, b):  # 计算两个向量的叉积
        a, b = torch.broadcast_tensors(a, b)
        return torch.cross(a, b, dim=-1)

    def th_norm(x, eps: float = 1e-8):  # 计算向量的欧几里得长度
        return x.square().sum(-1, keepdim=True).add(eps).sqrt()

    def th_N(x, alpha: float = 0):  # 向量归一化。将向量方向保留，长度化为1
        return x / th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab), th_N(bc), th_N(cd)  # 归一化向量
    # 计算法向量
    n1 = th_N(th_cross(ab, bc))
    n2 = th_N(th_cross(bc, cd))
    # 计算角度分量
    sin_angle = (th_cross(n1, bc) * n2).sum(-1)
    cos_angle = (n1 * n2).sum(-1)
    dih = torch.stack((cos_angle, sin_angle), -1)
    return dih

# 计算由四个原子A,B,C,D形成的两个平面（平面ABC和平面BCD）之间的夹角
def th_dih(a, b, c, d):
    return th_dih_v(a-b, b-c, c-d)

# 根据给定的两个向量X和Y，构造一个标准正交基（旋转矩阵）
def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)  # 归一化处理
    # 格拉姆-施密特正交化
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    # 利用叉积计算出同时垂直于Xn和Yn的向量
    Z = torch.cross(Xn, Yn, dim=-1)
    Zn =  Z / torch.linalg.norm(Z)
    return torch.stack((Xn, Yn, Zn), dim=-1)

RES_NB_JUMP = 50  # 链间残基编号的跳变

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL',
    'UNK','MAS',
    'HIS_D',  # 组氨酸的一种特定质子化状态或互变异构体，仅用于cart_bonded（立场计算）
    'Al', 'As', 'Au', 'B',
    'Be', 'Br', 'C', 'Ca', 'Cl',
    'Co', 'Cr', 'Cu', 'F', 'Fe',
    'Hg', 'I', 'Ir', 'K', 'Li', 'Mg',
    'Mn', 'Mo', 'N', 'Ni', 'O',
    'Os', 'P', 'Pb', 'Pd', 'Pr',
    'Pt', 'Re', 'Rh', 'Ru', 'S',
    'Sb', 'Se', 'Si', 'Sn', 'Tb',
    'Te', 'U', 'W', 'V', 'Y', 'Zn',
    'ATM'
]  # “ATM"是一个元素占位符，在生成时可以直接初始化为ATM

NAATOKENS = 20 + 2 + 1 + 46 + 1  # 20 aa + unk + mask + HIS_D + 46 atoms + ATM
NPROTAAS = 22  # 20aa + unk + mask + HIS_D。从0开始

NNAPROTAAS = 22  # 氨基酸和元素的token分界线。从0开始
NO_BOND = 0  # 无化学键链接
SINGLE_BOND = 1  # 单键
DOUBLE_BOND = 2  # 双键
TRIPLE_BOND = 3  # 三键
AROMATIC_BOND = 4  # 芳香键
RESIDUE_BB_BOND = 5  # 残基-残基连接键
RESIDUE_ATOM_BOND = 6  # 残基-配体连接键

NPROTANGS = 3  # 蛋白质主链的固定键角数量
NPROTTORS = 7  # 蛋白质的扭转角数量（通常包括主链的phi,psi,omega以及最多4个侧链chi角）
NTOTALTORS = 7  # 总扭转角数，这里直接继承RFAA的定义，但是不使用核酸，因此直接就是蛋白质的扭转角数量
NTOTALDOFS = NPROTTORS + NPROTANGS  # 总自由度

num2btype = [0, 1, 2, 3, 4,  # 没有键连接, 单键, 双键, 三键, 芳香键,
             5, 6]  # 蛋白质(多肽)残基-蛋白质(多肽)残基骨架键, 蛋白质(多肽)-配体原子键（共价键）, 未知键（初始状态）
NBTYPES = len(num2btype)

aa2num= {x:i for i,x in enumerate(num2aa)}  # token映射表

one_letter = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
              "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X"]

# 来自RFAA的原子框架优先级（与RFAA的表S10的顺序相反）
frame_priority2atom = ["F",  "Cl", "Br", "I",  "O",  "S",  "Se", "Te", "N",  "P", "As", "Sb",
                       "C",  "Si", "Sn", "Pb", "B",  "Al", "Zn", "Hg", "Cu", "Au", "Ni", "Pd",
                       "Pt", "Co", "Rh", "Ir", "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", "Mo",
                       "W",  "V",  "U",  "Tb", "Y",  "Be", "Mg", "Ca", "Li", "K",  "ATM"]

# 按原子框架优先级顺序排列的元素的原子序数
atom_num = [9,    17,   35,   53,   8,    16,   34,   52,   7,    15,   33,   51,
            6,    14,   32,   50,   82,   5,    13,   30,   80,   29,   79,   28,
            46,   78,   27,   45,   77,   26,   44,   76,   25,   75,   24,   42,
            23,   74,   92,   65,   39,   4,    12,   20,   3,    19,   0]

atom2frame_priority = {x: i for i, x in enumerate(frame_priority2atom)}  # 元素类型和框架优先级的映射
atomnum2atomtype = dict(zip(atom_num, frame_priority2atom))  # 元素类型和原子序数的字典
atomtype2atomnum = {v:k for k,v in atomnum2atomtype.items()}  # 原子序数和元素类型的字典

"""主链原子位置索引"""
class BBHeavyAtom(enum.IntEnum):
    N = 0; CA = 1; C = 2; O = 3; CB = 4; OXT = 14

# 20种常见氨基酸的原子手性信息
aachirals = [
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # ala
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # arg
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # asn
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # asp
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # cys
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # gln
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # glu
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # gly
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # his
    (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # ileu
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # leu
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # lys
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # met
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # phe
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # pro
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # ser
    (0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # thr
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # trp
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # tyr
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # val
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # unk
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # mas
]
aachirals = torch.tensor(aachirals)

# 原子名称列表（全原子表示）
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #0  ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), #1  arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), #2  asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #3  asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), #4  cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), #5  gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), #6  glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None," H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), #7  gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), #8  his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), #9  ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), #10 leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), #11 lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), #12 met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," H  "," HA ","1HB ","2HB ","1HD ","2HD ","1HE ","2HE "," HZ ",  None,  None,  None,  None), #13 phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), #14 pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None," H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), #15 ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), #16 thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"," H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), #17 trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," H  "," HA ","1HB ","2HB ","1HD ","1HE ","2HE ","2HD "," HH ",  None,  None,  None,  None), #18 tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None," H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), #19 val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #20 unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), #21 mask
    (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None," H  "," HA ","1HB ","2HB ","2HD ","1HE ","1HD ",  None,  None,  None,  None,  None,  None), #-1 his_d
]

# 原子名称列表（全原子表示，不带空格）
aa2long_noblank = [
    ("N", "CA", "C", "O", "CB", None,  None,  None,  None,  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "3HB", None, None, None, None, None, None, None, None),  # 0ala
    ("N", "CA", "C", "O", "CB", "CG",  "CD",  "NE",  "CZ",  "NH1", "NH2", None,  None,  None,  "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HD", "2HD", "HE", "1HH1", "2HH1", "1HH2", "2HH2"), # 1arg
    ("N", "CA", "C", "O", "CB", "CG",  "OD1", "ND2", None,  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "1HD2", "2HD2", None, None, None, None, None, None, None),  # 2asn
    ("N", "CA", "C", "O", "CB", "CG",  "OD1", "OD2", None,  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", None, None, None, None, None, None, None, None, None),  # 3asp
    ("N", "CA", "C", "O", "CB", "SG",  None,  None,  None,  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "HG", None, None, None, None, None, None, None, None),  # 4cys
    ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "NE2", None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HE2", "2HE2", None, None, None, None, None),  # 5gln
    ("N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "OE2", None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "1HG", "2HG", None, None, None, None, None, None, None),  # 6glu
    ("N", "CA", "C", "O", None, None,  None,  None,  None,  None,  None,  None,  None,  None,  "H", "1HA", "2HA", None, None, None, None, None, None, None, None, None, None),  # 7gly
    ("N", "CA", "C", "O", "CB", "CG",  "ND1", "CD2", "CE1", "NE2", None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "2HD", "1HE", "2HE", None, None, None, None, None, None),  # 8his
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", None,  None,  None,  None,  None,  None,  "H", "HA", "HB", "1HG2", "2HG2", "3HG2", "1HG1", "2HG1", "1HD1", "2HD1", "3HD1", None, None),  # 9ile
    ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", None,  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "HG", "1HD1", "2HD1", "3HD1", "1HD2", "2HD2", "3HD2", None, None),  # 10leu
    ("N", "CA", "C", "O", "CB", "CG",  "CD",  "CE",  "NZ",  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HD", "2HD", "1HE", "2HE", "1HZ", "2HZ", "3HZ"),  # 11lys
    ("N", "CA", "C", "O", "CB", "CG",  "SD",  "CE",  None,  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "1HG", "2HG", "1HE", "2HE", "3HE", None, None, None, None),  # 12met
    ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  None,  None,  None,  "H", "HA", "1HB", "2HB", "1HD", "2HD", "1HE", "2HE", "HZ", None, None, None, None),  # 13phe
    ("N", "CA", "C", "O", "CB", "CG",  "CD",  None,  None,  None,  None,  None,  None,  None,  "HA", "1HB", "2HB", "1HG", "2HG", "1HD", "2HD", None, None, None, None, None, None),  # 14pro
    ("N", "CA", "C", "O", "CB", "OG",  None,  None,  None,  None,  None,  None,  None,  None,  "H", "HG", "HA", "1HB", "2HB", None, None, None, None, None, None, None, None),  # 15ser
    ("N", "CA", "C", "O", "CB", "OG1", "CG2", None,  None,  None,  None,  None,  None,  None,  "H", "HG1", "HA", "HB", "1HG2", "2HG2", "3HG2", None, None, None, None, None, None),  # 16thr
    ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2", "H", "HA", "1HB", "2HB", "1HD", "1HE", "HZ2", "HH2", "HZ3", "HE3", None, None, None),  # 17trp
    ("N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "OH",  None,  None,  "H", "HA", "1HB", "2HB", "1HD", "1HE", "2HE", "2HD", "HH", None, None, None, None),  # 18tyr
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", None,  None,  None,  None,  None,  None,  None,  "H", "HA", "HB", "1HG1", "2HG1", "3HG1", "1HG2", "2HG2", "3HG2", None, None, None, None),  # 19val
    ("N", "CA", "C", "O", "CB", None,  None,  None,  None,  None,  None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "3HB", None, None, None, None, None, None, None, None),  # 20unk
    ("N", "CA", "C", "O", "CB", "CG",  "NE2", "CD2", "CE1", "ND1", None,  None,  None,  None,  "H", "HA", "1HB", "2HB", "2HD", "1HE", "1HD", None, None, None, None, None, None), #-1 his_d
]

# 对称原子替换列表，用于处理侧链的对称性，防止模型因为侧链翻转（而在物理上是对的）受到不必要的惩罚
aa2longalt=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None,  " H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2"), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","1HD2","2HD2",  None,  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD2"," OD1",  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB "," HG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE2"," OE1",  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","1HG ","2HG ",  None,  None,  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  " H  ","1HA ","2HA ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","2HD ","1HE ","2HE ",  None,  None,  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None,  " H  "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ "), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ ",  None,  None,  None,  " H  ","2HD ","2HE "," HZ ","1HE ","1HD "," HA ","1HB ","2HB ",  None,  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None,  " HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None,  " H  "," HG "," HA ","1HB ","2HB ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  " H  "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2",  " H  "," HA ","1HB ","2HB ","1HD ","1HE "," HZ2"," HH2"," HZ3"," HE3",  None,  None,  None), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD2"," CD1"," CE2"," CE1"," CZ "," OH ",  None,  None,  " H  "," HA ","1HB ","2HB ","2HD ","2HE ","1HE ","1HD "," HH ",  None,  None,  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None,  " H  "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # unk
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  " H  "," HA ","1HB ","2HB ","3HB ",  None,  None,  None,  None,  None,  None,  None,  None), # mask
]

# 原子化学类型，定义了每个原子的化学性质。用于查找物理参数（范德华半径、电荷等）或者用于计算物理能量项（如 LJ 势能）
aa2type = [
    ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # ala
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CH2", "NtrR","aroC","Narg","Narg",  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol","Hpol","Hpol"), # arg
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CNH2","ONH2","NH2O",  None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hpol","Hpol",  None,  None,  None,  None,  None,  None,  None), # asn
    ("Nbb", "CAbb","CObb","OCbb","CH2", "COO", "OOC", "OOC",   None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None,  None), # asp
    ("Nbb", "CAbb","CObb","OCbb","CH2", "SH1",   None,  None,  None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","HS",    None,  None,  None,  None,  None,  None,  None,  None), # cys
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CNH2","ONH2","NH2O",  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol",  None,  None,  None,  None,  None), # gln
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "COO", "OOC", "OOC",   None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None), # glu
    ("Nbb", "CAbb","CObb","OCbb",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "Nhis","aroC","aroC","Ntrp",  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hpol","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # his
    ("Nbb", "CAbb","CObb","OCbb","CH1", "CH2", "CH3", "CH3",   None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None), # ile
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH1", "CH3", "CH3",   None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None), # leu
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "CH2", "CH2", "Nlys",  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hpol","Hpol","Hpol"), # lys
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH2", "S",   "CH3",   None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None), # met
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","aroC","aroC","aroC","aroC",  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Haro","Haro","Haro","Haro","Haro",  None,  None,  None,  None), # phe
    ("Npro","CAbb","CObb","OCbb","CH2", "CH2", "CH2",   None,  None,  None,  None,  None,  None,  None, "Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # pro
    ("Nbb", "CAbb","CObb","OCbb","CH2", "OH",    None,  None,  None,  None,  None,  None,  None,  None, "HNbb","Hpol","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    ("Nbb", "CAbb","CObb","OCbb","CH1", "OH",  "CH3",   None,  None,  None,  None,  None,  None,  None, "HNbb","Hpol","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None), # thr
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","CH0", "CH0", "aroC","Ntrp","aroC","aroC","aroC", "HNbb","Haro","Hapo","Hapo","Hapo","Hpol","Haro","Haro","Haro","Haro",  None,  None,  None), # trp
    ("Nbb", "CAbb","CObb","OCbb","CH2", "CH0", "aroC","aroC","aroC","aroC","CH0", "OHY",   None,  None, "HNbb","Haro","Haro","Haro","Haro","Hapo","Hapo","Hapo","Hpol",  None,  None,  None,  None), # tyr
    ("Nbb", "CAbb","CObb","OCbb","CH1", "CH3", "CH3",   None,  None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None), # val
    ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # unk
    ("Nbb", "CAbb","CObb","OCbb","CH3",   None,  None,  None,  None,  None,  None,  None,  None,  None, "HNbb","Hapo","Hapo","Hapo","Hapo",  None,  None,  None,  None,  None,  None,  None,  None), # mask
    (None, "genAl", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Al
    (None, "genAs", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # As
    (None, "genAu", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Au
    (None, "genB",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # B
    (None, "genBe", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Be
    (None, "genBr", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Br
    (None, "genC",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # C
    (None, "genCa", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ca
    (None, "genCl", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cl
    (None, "genCo", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Co
    (None, "genCr", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cr
    (None, "genCu", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Cu
    (None, "genF",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # F
    (None, "genFe", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Fe
    (None, "genHg", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Hg
    (None, "genI",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # I
    (None, "genIr", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ir
    (None, "genK",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # K
    (None, "genLi", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Li
    (None, "genMg", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mg
    (None, "genMn", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mn
    (None, "genMo", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Mo
    (None, "genN",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # N
    (None, "genNi", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ni
    (None, "genO",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # O
    (None, "genOs", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Os
    (None, "genP",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # P
    (None, "genPb", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pb
    (None, "genPd", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pd
    (None, "genPr", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pr
    (None, "genPt", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Pt
    (None, "genRe", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Re
    (None, "genRh", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Rh
    (None, "genRu", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Ru
    (None, "genS",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # S
    (None, "genSb", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Sb
    (None, "genSe", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Se
    (None, "genSi", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Si
    (None, "genSn", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Sn
    (None, "genTb", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Tb
    (None, "genTe", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Te
    (None, "genU",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # U
    (None, "genW",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # W
    (None, "genV",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # V
    (None, "genY",  None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Y
    (None, "genZn", None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # Zn
    (None, "genATM",None,  None,  None,   None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # ATM
]


# 简单地列出每个位置是碳（C）、氮（N）、氧（O）等元素
aa2elt = [
    ("N","C","C","O","C",None,None,None,None,None,None,None,None,None, "H","H","H","H","H",None,None,None,None,None,None,None,None), # ala
    ("N","C","C","O","C","C","C","N","C","N","N",None,None,None, "H","H","H","H","H","H","H","H","H","H","H","H","H"), # arg
    ("N","C","C","O","C","C","O","N",None,None,None,None,None,None, "H","H","H","H","H","H",None,None,None,None,None,None,None), # asn
    ("N","C","C","O","C","C","O","O",None,None,None,None,None,None, "H","H","H","H",None,None,None,None,None,None,None,None,None), # asp
    ("N","C","C","O","C","S",None,None,None,None,None,None,None,None, "H","H","H","H","H",None,None,None,None,None,None,None,None), # cys
    ("N","C","C","O","C","C","C","O","N",None,None,None,None,None, "H","H","H","H","H","H","H","H",None,None,None,None,None), # gln
    ("N","C","C","O","C","C","C","O","O",None,None,None,None,None, "H","H","H","H","H","H",None,None,None,None,None,None,None), # glu
    ("N","C","C","O",None,None,None,None,None,None,None,None,None,None, "H","H","H",None,None,None,None,None,None,None,None,None,None), # gly
    ("N","C","C","O","C","C","N","C","C","N",None,None,None,None, "H","H","H","H","H","H","H",None,None,None,None,None,None), # his
    ("N","C","C","O","C","C","C","C",None,None,None,None,None,None, "H","H","H","H","H","H","H","H","H","H","H",None,None), # ile
    ("N","C","C","O","C","C","C","C",None,None,None,None,None,None, "H","H","H","H","H","H","H","H","H","H","H",None,None), # leu
    ("N","C","C","O","C","C","C","C","N",None,None,None,None,None, "H","H","H","H","H","H","H","H","H","H","H","H","H"), # lys
    ("N","C","C","O","C","C","S","C",None,None,None,None,None,None, "H","H","H","H","H","H","H","H","H",None,None,None,None), # met
    ("N","C","C","O","C","C","C","C","C","C","C",None,None,None, "H","H","H","H","H","H","H","H","H",None,None,None,None), # phe
    ("N","C","C","O","C","C","C",None,None,None,None,None,None,None, "H","H","H","H","H","H","H",None,None,None,None,None,None), # pro
    ("N","C","C","O","C","O",None,None,None,None,None,None,None,None, "H","H","H","H","H",None,None,None,None,None,None,None,None), # ser
    ("N","C","C","O","C","O","C",None,None,None,None,None,None,None, "H","H","H","H","H","H","H",None,None,None,None,None,None), # thr
    ("N","C","C","O","C","C","C","C","C","C","N","C","C","C", "H","H","H","H","H","H","H","H","H","H",None,None,None), # trp
    ("N","C","C","O","C","C","C","C","C","C","C","O",None,None, "H","H","H","H","H","H","H","H","H",None,None,None,None), # tyr
    ("N","C","C","O","C","C","C",None,None,None,None,None,None,None, "H","H","H","H","H","H","H","H","H",None,None,None,None), # val
    ("N","C","C","O","C",None,None,None,None,None,None,None,None,None, "H","H","H","H","H",None,None,None,None,None,None,None,None), # unk
    ("N","C","C","O","C",None,None,None,None,None,None,None,None,None, "H","H","H","H","H",None,None,None,None,None,None,None,None), # mask
]

# 刚体坐标系的定义。格式为[原点原子, X轴方向原子, XY平面原子]
frames=[
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # ala
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "], [" CG "," CD "," NE "] ],  # arg
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # asn
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # asp
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "] ],  # cys
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "] ],  # gln
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "] ],  # glu
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # gly
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # his
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG1"] ],  # ile
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # leu
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "], [" CG "," CD "," CE "] ],  # lys
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," SD "] ],  # met
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # phe
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "], [" CB "," CG "," CD "]],  # pro
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," OG "] ],  # ser
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," OG1"] ],  # thr
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # trp
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "], [" CA "," CB "," CG "] ],  # tyr
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "],[" N  "," CA "," CB "] ],  # val
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # unk
    [ [" N  "," CA "," C  "],[" CA "," C  "," O  "] ],  # mask
]

# 化学键连接图的列表形式
aabonds=[
    #       0               1               2                3               4              5               6               7               8               9              10              11              12              13              14              15              16              17              18              19              20              21              22               23             24
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB ","1HB "),(" CB ","2HB "),(" CB ","3HB ")) , # ala
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," NE "),(" CD ","1HD "),(" CD ","2HD "),(" NE "," CZ "),(" NE "," HE "),(" CZ "," NH1"),(" CZ "," NH2"),(" NH1","1HH1"),(" NH1","2HH1"),(" NH2","1HH2"),(" NH2","2HH2")) , # arg
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," OD1"),(" CG "," ND2"),(" ND2","1HD2"),(" ND2","2HD2")) , # asn
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," OD1"),(" CG "," OD2")) , # asp
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," SG "),(" CB ","1HB "),(" CB ","2HB "),(" SG "," HG ")) , # cys
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," OE1"),(" CD "," NE2"),(" NE2","1HE2"),(" NE2","2HE2")) , # gln
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," OE1"),(" CD "," OE2")) , # glu
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA ","1HA "),(" CA ","2HA "),(" C  "," O  ")) , # gly
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," ND1"),(" CG "," CD2"),(" ND1"," CE1"),(" CD2"," NE2"),(" CD2","2HD "),(" CE1"," NE2"),(" CE1","1HE "),(" NE2","2HE ")) , # his
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG1"),(" CB "," CG2"),(" CB "," HB "),(" CG1"," CD1"),(" CG1","1HG1"),(" CG1","2HG1"),(" CG2","1HG2"),(" CG2","2HG2"),(" CG2","3HG2"),(" CD1","1HD1"),(" CD1","2HD1"),(" CD1","3HD1")) , # ile
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CG "," HG "),(" CD1","1HD1"),(" CD1","2HD1"),(" CD1","3HD1"),(" CD2","1HD2"),(" CD2","2HD2"),(" CD2","3HD2")) , # leu
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD "," CE "),(" CD ","1HD "),(" CD ","2HD "),(" CE "," NZ "),(" CE ","1HE "),(" CE ","2HE "),(" NZ ","1HZ "),(" NZ ","2HZ "),(" NZ ","3HZ ")) , # lys
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," SD "),(" CG ","1HG "),(" CG ","2HG "),(" SD "," CE "),(" CE ","1HE "),(" CE ","2HE "),(" CE ","3HE ")) , # met
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CD1"," CE1"),(" CD1","1HD "),(" CD2"," CE2"),(" CD2","2HD "),(" CE1"," CZ "),(" CE1","1HE "),(" CE2"," CZ "),(" CE2","2HE "),(" CZ "," HZ ")) , # phe
    ((" N  "," CA "),(" N  "," CD "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD "),(" CG ","1HG "),(" CG ","2HG "),(" CD ","1HD "),(" CD ","2HD ")) , # pro
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," OG "),(" CB ","1HB "),(" CB ","2HB "),(" OG "," HG ")) , # ser
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," OG1"),(" CB "," CG2"),(" CB "," HB "),(" OG1"," HG1"),(" CG2","1HG2"),(" CG2","2HG2"),(" CG2","3HG2")) , # thr
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CD1"," NE1"),(" CD1","1HD "),(" CD2"," CE2"),(" CD2"," CE3"),(" NE1"," CE2"),(" NE1","1HE "),(" CE2"," CZ2"),(" CE3"," CZ3"),(" CE3"," HE3"),(" CZ2"," CH2"),(" CZ2"," HZ2"),(" CZ3"," CH2"),(" CZ3"," HZ3"),(" CH2"," HH2")) , # trp
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG "),(" CB ","1HB "),(" CB ","2HB "),(" CG "," CD1"),(" CG "," CD2"),(" CD1"," CE1"),(" CD1","1HD "),(" CD2"," CE2"),(" CD2","2HD "),(" CE1"," CZ "),(" CE1","1HE "),(" CE2"," CZ "),(" CE2","2HE "),(" CZ "," OH "),(" OH "," HH ")) , # tyr
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB "," CG1"),(" CB "," CG2"),(" CB "," HB "),(" CG1","1HG1"),(" CG1","2HG1"),(" CG1","3HG1"),(" CG2","1HG2"),(" CG2","2HG2"),(" CG2","3HG2")), # val
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB ","1HB "),(" CB ","2HB "),(" CB ","3HB ")) , # unk
    ((" N  "," CA "),(" N  "," H  "),(" CA "," C  "),(" CA "," CB "),(" CA "," HA "),(" C  "," O  "),(" CB ","1HB "),(" CB ","2HB "),(" CB ","3HB ")) , # mask
]

# 氨基酸全原子理想坐标
ideal_coords = [
    [ # 0 ala
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
        [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
        ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
        ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
        ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
    ],
    [ # 1 arg
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3467, -0.5055,  0.9018)],
        [' CB ', 8, (-0.5042,-0.7698,-1.2118)],
        ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
        ['2HB ', 4, ( 0.3639, -0.5323, -0.8789)],
        [' CG ', 4, (0.6396,1.3794, 0.000)],
        ['1HG ', 5, (0.3639, -0.5139,  0.8900)],
        ['2HG ', 5, (0.3641, -0.5140, -0.8903)],
        [' CD ', 5, (0.5492,1.3801, 0.000)],
        ['1HD ', 6, (0.3637, -0.5135,  0.8895)],
        ['2HD ', 6, (0.3636, -0.5134, -0.8893)],
        [' NE ', 6, (0.5423,1.3491, 0.000)],
        [' NH1', 7, (0.2012,2.2965, 0.000)],
        [' NH2', 7, (2.0824,1.0030, 0.000)],
        [' CZ ', 7, (0.7650,1.1090, 0.000)],
        [' HE ', 7, (0.4701,-0.8955, 0.000)],
        ['1HH1', 7, (-0.8059,2.3776, 0.000)],
        ['1HH2', 7, (2.5160,0.0898, 0.000)],
        ['2HH1', 7, (0.7745,3.1277, 0.000)],
        ['2HH2', 7, (2.6554,1.8336, 0.000)],
    ],
    [ # 2 asn
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3233, -0.4967,  0.9162)],
        [' CB ', 8, (-0.5341,-0.7799,-1.1874)],
        ['1HB ', 4, ( 0.3641, -0.5327,  0.8795)],
        ['2HB ', 4, ( 0.3639, -0.5323, -0.8789)],
        [' CG ', 4, (0.5778,1.3881, 0.000)],
        [' ND2', 5, (0.5839,-1.1711, 0.000)],
        [' OD1', 5, (0.6331,1.0620, 0.000)],
        ['1HD2', 5, (1.5825, -1.2322, 0.000)],
        ['2HD2', 5, (0.0323, -2.0046, 0.000)],
    ],
    [ # 3 asp
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3233, -0.4967,  0.9162)],
        [' CB ', 8, (-0.5162,-0.7757,-1.2144)],
        ['1HB ', 4, ( 0.3639, -0.5324,  0.8791)],
        ['2HB ', 4, ( 0.3640, -0.5325, -0.8792)],
        [' CG ', 4, (0.5926,1.4028, 0.000)],
        [' OD1', 5, (0.5746,1.0629, 0.000)],
        [' OD2', 5, (0.5738,-1.0627, 0.000)],
    ],
    [ # 4 cys
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3481, -0.5059,  0.9006)],
        [' CB ', 8, (-0.5046,-0.7727,-1.2189)],
        ['1HB ', 4, ( 0.3639, -0.5324,  0.8791)],
        ['2HB ', 4, ( 0.3638, -0.5322, -0.8787)],
        [' SG ', 4, (0.7386,1.6511, 0.000)],
        [' HG ', 5, (0.1387,1.3221, 0.000)],
    ],
    [ # 5 gln
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3363, -0.5013,  0.9074)],
        [' CB ', 8, (-0.5226,-0.7776,-1.2109)],
        ['1HB ', 4, ( 0.3638, -0.5323,  0.8789)],
        ['2HB ', 4, ( 0.3638, -0.5322, -0.8788)],
        [' CG ', 4, (0.6225,1.3857, 0.000)],
        ['1HG ', 5, ( 0.3531, -0.5156,  0.8931)],
        ['2HG ', 5, ( 0.3531, -0.5156, -0.8931)],
        [' CD ', 5, (0.5788,1.4021, 0.000)],
        [' NE2', 6, (0.5908,-1.1895, 0.000)],
        [' OE1', 6, (0.6347,1.0584, 0.000)],
        ['1HE2', 6, (1.5825, -1.2525, 0.000)],
        ['2HE2', 6, (0.0380, -2.0229, 0.000)],
    ],
    [ # 6 glu
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3363, -0.5013,  0.9074)],
        [' CB ', 8, (-0.5197,-0.7737,-1.2137)],
        ['1HB ', 4, ( 0.3638, -0.5323,  0.8789)],
        ['2HB ', 4, ( 0.3638, -0.5322, -0.8788)],
        [' CG ', 4, (0.6287,1.3862, 0.000)],
        ['1HG ', 5, ( 0.3531, -0.5156,  0.8931)],
        ['2HG ', 5, ( 0.3531, -0.5156, -0.8931)],
        [' CD ', 5, (0.5850,1.3849, 0.000)],
        [' OE1', 6, (0.5752,1.0618, 0.000)],
        [' OE2', 6, (0.5741,-1.0635, 0.000)],
    ],
    [ # 7 gly
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        ['1HA ', 0, ( -0.3676, -0.5329,  0.8771)],
        ['2HA ', 0, ( -0.3674, -0.5325, -0.8765)],
    ],
    [ # 8 his
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3299, -0.5180,  0.9001)],
        [' CB ', 8, (-0.5163,-0.7809,-1.2129)],
        ['1HB ', 4, ( 0.3640, -0.5325,  0.8793)],
        ['2HB ', 4, ( 0.3637, -0.5321, -0.8786)],
        [' CG ', 4, (0.6016,1.3710, 0.000)],
        [' CD2', 5, (0.8918,-1.0184, 0.000)],
        [' CE1', 5, (2.0299,0.8564, 0.000)],
        ['1HE ', 5, (2.8542, 1.5693,  0.000)],
        ['2HD ', 5, ( 0.6584, -2.0835, 0.000) ],
        [' ND1', 6, (-1.8631, -1.0722,  0.000)],
        [' NE2', 6, (-1.8625,  1.0707, 0.000)],
        ['2HE ', 6, (-1.5439,  2.0292, 0.000)],
    ],
    [ # 9 ile
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3405, -0.5028,  0.9044)],
        [' CB ', 8, (-0.5140,-0.7885,-1.2184)],
        [' HB ', 4, (0.3637, -0.4714,  0.9125)],
        [' CG1', 4, (0.5339,1.4348,0.000)],
        [' CG2', 4, (0.5319,-0.7693,-1.1994)],
        ['1HG2', 4, (1.6215, -0.7588, -1.1842)],
        ['2HG2', 4, (0.1785, -1.7986, -1.1569)],
        ['3HG2', 4, (0.1773, -0.3016, -2.1180)],
        [' CD1', 5, (0.6106,1.3829, 0.000)],
        ['1HG1', 5, (0.3637, -0.5338,  0.8774)],
        ['2HG1', 5, (0.3640, -0.5322, -0.8793)],
        ['1HD1', 5, (1.6978,  1.3006, 0.000)],
        ['2HD1', 5, (0.2873,  1.9236, -0.8902)],
        ['3HD1', 5, (0.2888, 1.9224, 0.8896)],
    ],
    [ # 10 leu
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.525, -0.000, -0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3435, -0.5040,  0.9027)],
        [' CB ', 8, (-0.5175,-0.7692,-1.2220)],
        ['1HB ', 4, ( 0.3473, -0.5346,  0.8827)],
        ['2HB ', 4, ( 0.3476, -0.5351, -0.8836)],
        [' CG ', 4, (0.6652,1.3823, 0.000)],
        [' CD1', 5, (0.5083,1.4353, 0.000)],
        [' CD2', 5, (0.5079,-0.7600,1.2163)],
        [' HG ', 5, (0.3640, -0.4825, -0.9075)],
        ['1HD1', 5, (1.5984,  1.4353, 0.000)],
        ['2HD1', 5, (0.1462,  1.9496, -0.8903)],
        ['3HD1', 5, (0.1459, 1.9494, 0.8895)],
        ['1HD2', 5, (1.5983, -0.7606,  1.2158)],
        ['2HD2', 5, (0.1456, -0.2774,  2.1243)],
        ['3HD2', 5, (0.1444, -1.7871,  1.1815)],
    ],
    [ # 11 lys
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3335, -0.5005,  0.9097)],
        ['1HB ', 4, ( 0.3640, -0.5324,  0.8791)],
        ['2HB ', 4, ( 0.3639, -0.5324, -0.8790)],
        [' CB ', 8, (-0.5259,-0.7785,-1.2069)],
        ['1HG ', 5, (0.3641, -0.5229,  0.8852)],
        ['2HG ', 5, (0.3637, -0.5227, -0.8841)],
        [' CG ', 4, (0.6291,1.3869, 0.000)],
        [' CD ', 5, (0.5526,1.4174, 0.000)],
        ['1HD ', 6, (0.3641, -0.5239,  0.8848)],
        ['2HD ', 6, (0.3638, -0.5219, -0.8850)],
        [' CE ', 6, (0.5544,1.4170, 0.000)],
        [' NZ ', 7, (0.5566,1.3801, 0.000)],
        ['1HE ', 7, (0.4199, -0.4638,  0.9482)],
        ['2HE ', 7, (0.4202, -0.4631, -0.8172)],
        ['1HZ ', 7, (1.6223, 1.3980, 0.0658)],
        ['2HZ ', 7, (0.2970,  1.9326, -0.7584)],
        ['3HZ ', 7, (0.2981, 1.9319, 0.8909)],
    ],
    [ # 12 met
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3303, -0.4990,  0.9108)],
        ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
        ['2HB ', 4, ( 0.3641, -0.5326, -0.8795)],
        [' CB ', 8, (-0.5331,-0.7727,-1.2048)],
        ['1HG ', 5, (0.3637, -0.5256,  0.8823)],
        ['2HG ', 5, (0.3638, -0.5249, -0.8831)],
        [' CG ', 4, (0.6298,1.3858,0.000)],
        [' SD ', 5, (0.6953,1.6645,0.000)],
        [' CE ', 6, (0.3383,1.7581,0.000)],
        ['1HE ', 6, (1.7054,  2.0532, -0.0063)],
        ['2HE ', 6, (0.1906,  2.3099, -0.9072)],
        ['3HE ', 6, (0.1917, 2.3792, 0.8720)],
    ],
    [ # 13 phe
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3303, -0.4990,  0.9108)],
        ['1HB ', 4, ( 0.3635, -0.5318,  0.8781)],
        ['2HB ', 4, ( 0.3641, -0.5326, -0.8795)],
        [' CB ', 8, (-0.5150,-0.7729,-1.2156)],
        [' CG ', 4, (0.6060,1.3746, 0.000)],
        [' CD1', 5, (0.7078,1.1928, 0.000)],
        [' CD2', 5, (0.7084,-1.1920, 0.000)],
        [' CE1', 5, (2.0900,1.1940, 0.000)],
        [' CE2', 5, (2.0897,-1.1939, 0.000)],
        [' CZ ', 5, (2.7809, 0.000, 0.000)],
        ['1HD ', 5, (0.1613, 2.1362, 0.000)],
        ['2HD ', 5, (0.1621, -2.1360, 0.000)],
        ['1HE ', 5, (2.6335,  2.1384, 0.000)],
        ['2HE ', 5, (2.6344, -2.1378, 0.000)],
        [' HZ ', 5, (3.8700, 0.000, 0.000)],
    ],
    [ # 14 pro
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' HA ', 0, (-0.3868, -0.5380,  0.8781)],
        ['1HB ', 4, ( 0.3762, -0.5355,  0.8842)],
        ['2HB ', 4, ( 0.3762, -0.5355, -0.8842)],
        [' CB ', 8, (-0.5649,-0.5888,-1.2966)],
        [' CG ', 4, (0.3657,1.4451,0.0000)],
        [' CD ', 5, (0.3744,1.4582, 0.0)],
        ['1HG ', 5, (0.3798, -0.5348,  0.8830)],
        ['2HG ', 5, (0.3798, -0.5348, -0.8830)],
        ['1HD ', 6, (0.3798, -0.5348,  0.8830)],
        ['2HD ', 6, (0.3798, -0.5348, -0.8830)],
    ],
    [ # 15 ser
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3425, -0.5041,  0.9048)],
        ['1HB ', 4, ( 0.3637, -0.5321,  0.8786)],
        ['2HB ', 4, ( 0.3636, -0.5319, -0.8782)],
        [' CB ', 8, (-0.5146,-0.7595,-1.2073)],
        [' OG ', 4, (0.5021,1.3081, 0.000)],
        [' HG ', 5, (0.2647, 0.9230, 0.000)],
    ],
    [ # 16 thr
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3364, -0.5015,  0.9078)],
        [' HB ', 4, ( 0.3638, -0.5006,  0.8971)],
        ['1HG2', 4, ( 1.6231, -0.7142, -1.2097)],
        ['2HG2', 4, ( 0.1792, -1.7546, -1.2237)],
        ['3HG2', 4, ( 0.1808, -0.2222, -2.1269)],
        [' CB ', 8, (-0.5172,-0.7952,-1.2130)],
        [' CG2', 4, (0.5334,-0.7239,-1.2267)],
        [' OG1', 4, (0.4804,1.3506,0.000)],
        [' HG1', 5, (0.3194,  0.9056, 0.000)],
    ],
    [ # 17 trp
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3436, -0.5042,  0.9031)],
        ['1HB ', 4, ( 0.3639, -0.5323,  0.8790)],
        ['2HB ', 4, ( 0.3638, -0.5322, -0.8787)],
        [' CB ', 8, (-0.5136,-0.7712,-1.2173)],
        [' CG ', 4, (0.5984,1.3741, 0.000)],
        [' CD1', 5, (0.8151,1.0921, 0.000)],
        [' CD2', 5, (0.8753,-1.1538, 0.000)],
        [' CE2', 5, (2.1865,-0.6707, 0.000)],
        [' CE3', 5, (0.6541,-2.5366, 0.000)],
        [' NE1', 5, (2.1309,0.7003, 0.000)],
        [' CH2', 5, (3.0315,-2.8930, 0.000)],
        [' CZ2', 5, (3.2813,-1.5205, 0.000)],
        [' CZ3', 5, (1.7521,-3.3888, 0.000)],
        ['1HD ', 5, (0.4722, 2.1252,  0.000)],
        ['1HE ', 5, ( 2.9291,  1.3191,  0.000)],
        [' HE3', 5, (-0.3597, -2.9356,  0.000)],
        [' HZ2', 5, (4.3053, -1.1462,  0.000)],
        [' HZ3', 5, ( 1.5712, -4.4640,  0.000)],
        [' HH2', 5, ( 3.8700, -3.5898,  0.000)],
    ],
    [ # 18 tyr
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3305, -0.4992,  0.9112)],
        ['1HB ', 4, ( 0.3642, -0.5327,  0.8797)],
        ['2HB ', 4, ( 0.3637, -0.5321, -0.8785)],
        [' CB ', 8, (-0.5305,-0.7799,-1.2051)],
        [' CG ', 4, (0.6104,1.3840, 0.000)],
        [' CD1', 5, (0.6936,1.2013, 0.000)],
        [' CD2', 5, (0.6934,-1.2011, 0.000)],
        [' CE1', 5, (2.0751,1.2013, 0.000)],
        [' CE2', 5, (2.0748,-1.2011, 0.000)],
        [' OH ', 5, (4.1408, 0.000, 0.000)],
        [' CZ ', 5, (2.7648, 0.000, 0.000)],
        ['1HD ', 5, (0.1485, 2.1455,  0.000)],
        ['2HD ', 5, (0.1484, -2.1451,  0.000)],
        ['1HE ', 5, (2.6200, 2.1450,  0.000)],
        ['2HE ', 5, (2.6199, -2.1453,  0.000)],
        [' HH ', 6, (0.3190, 0.9057,  0.000)],
    ],
    [ # 19 val
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3497, -0.5068,  0.9002)],
        [' CB ', 8, (-0.5105,-0.7712,-1.2317)],
        [' CG1', 4, (0.5326,1.4252, 0.000)],
        [' CG2', 4, (0.5177,-0.7693,1.2057)],
        [' HB ', 4, (0.3541, -0.4754, -0.9148)],
        ['1HG1', 4, (1.6228,  1.4063,  0.000)],
        ['2HG1', 4, (0.1790,  1.9457, -0.8898)],
        ['3HG1', 4, (0.1798, 1.9453, 0.8903)],
        ['1HG2', 4, (1.6073, -0.7659,  1.1989)],
        ['2HG2', 4, (0.1586, -0.2971,  2.1203)],
        ['3HG2', 4, (0.1582, -1.7976,  1.1631)],
    ],
    [ # 20 unk
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
        [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
        ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
        ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
        ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
    ],
    [ # 21 mask
        [' N  ', 0, (-0.5272, 1.3593, 0.000)],
        [' CA ', 0, (0.000, 0.000, 0.000)],
        [' C  ', 0, (1.5233, 0.000, 0.000)],
        [' O  ', 3, (0.6303, 1.0574, 0.000)],
        [' H  ', 2, (0.4920,-0.8821,  0.0000)],
        [' HA ', 0, (-0.3341, -0.4928,  0.9132)],
        [' CB ', 8, (-0.5289,-0.7734,-1.1991)],
        ['1HB ', 8, (-0.1265, -1.7863, -1.1851)],
        ['2HB ', 8, (-1.6173, -0.8147, -1.1541)],
        ['3HB ', 8, (-0.2229, -0.2744, -2.1172)],
    ],
]

# 组成氨基酸侧链扭转角的原子组合
torsions=[
    [ None, None, None, None ],  # ala
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," NE "], [" CG "," CD "," NE "," CZ "] ],  # arg
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," OD1"], None, None ],  # asn
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," OD1"], None, None ],  # asp
    [ [" N  "," CA "," CB "," SG "], [" CA "," CB "," SG "," HG "], None, None ],  # cys
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," OE1"], None ],  # gln
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," OE1"], None ],  # glu
    [ None, None, None, None ],  # gly
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," ND1"], [" CD2"," CE1","1HE "," NE2"], None ],  # his (protonation handled as a pseudo-torsion)
    [ [" N  "," CA "," CB "," CG1"], [" CA "," CB "," CG1"," CD1"], None, None ],  # ile
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # leu
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD "," CE "], [" CG "," CD "," CE "," NZ "] ],  # lys
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," SD "], [" CB "," CG "," SD "," CE "], None ],  # met
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # phe
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD "], [" CB "," CG "," CD ","1HD "], None ],  # pro
    [ [" N  "," CA "," CB "," OG "], [" CA "," CB "," OG "," HG "], None, None ],  # ser
    [ [" N  "," CA "," CB "," OG1"], [" CA "," CB "," OG1"," HG1"], None, None ],  # thr
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], None, None ],  # trp
    [ [" N  "," CA "," CB "," CG "], [" CA "," CB "," CG "," CD1"], [" CE1"," CZ "," OH "," HH "], None ],  # tyr
    [ [" N  "," CA "," CB "," CG1"], None, None, None ],  # val
    [ None, None, None, None ],  # unk
    [ None, None, None, None ],  # mask
]

NTOTAL = 27  # 蛋白质中每个残基最多的原子数量
# 主链理想坐标
init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
INIT_CRDS = torch.full((NTOTAL, 3), np.nan)
INIT_CRDS[:3] = torch.stack((init_N, init_CA, init_C), dim=0)

norm_N = init_N / (torch.norm(init_N, dim=-1, keepdim=True) + 1e-5)
norm_C = init_C / (torch.norm(init_C, dim=-1, keepdim=True) + 1e-5)
cos_ideal_NCAC = torch.sum(norm_N*norm_C, dim=-1)  # cosine of ideal N-CA-C bond angle

NFRAMES = max([len(f) for f in frames])

# 构建从完整表示（N×27）中的原子到“替代"表示的映射关系
allatom_mask = torch.zeros((NAATOKENS, NTOTAL), dtype=torch.bool)
long2alt = torch.zeros((NAATOKENS, NTOTAL), dtype=torch.long)
for i in range(NNAPROTAAS):
    i_l, i_lalt = aa2long[i], aa2longalt[i]
    for j,a in enumerate(i_l):
        if (a is None):
            long2alt[i,j] = j
        else:
            long2alt[i,j] = i_lalt.index(a)
            allatom_mask[i,j] = True
for i in range(NNAPROTAAS, NAATOKENS):
    for j in range(NTOTAL):
        long2alt[i, j] = j
allatom_mask[NNAPROTAAS:, 1] = True  # 使用特定的索引位置来表示其几何中心或关键原子

# 原子类型索引
idx2aatype = []
for x in aa2type:
    for y in x:
        if y and y not in idx2aatype:
            idx2aatype.append(y)
aatype2idx = {x:i for i,x in enumerate(idx2aatype)}

# 元素索引
idx2elt = []
for x in aa2elt:
    for y in x:
        if y and y not in idx2elt:
            idx2elt.append(y)
elt2idx = {x:i for i,x in enumerate(idx2elt)}

# LJ/LK势能评分参数
atom_type_index = torch.zeros((NAATOKENS, NTOTAL), dtype=torch.long)
element_index = torch.zeros((NAATOKENS, NTOTAL), dtype=torch.long)

ljlk_parameters = torch.zeros((NAATOKENS,NTOTAL,5), dtype=torch.float)
lj_correction_parameters = torch.zeros((NAATOKENS,NTOTAL,4), dtype=bool) # donor/acceptor/hpol/disulf
for i in range(NNAPROTAAS):
    for j,a in enumerate(aa2type[i]):
        if (a is not None):
            atom_type_index[i,j] = aatype2idx[a]
            ljlk_parameters[i,j,:] = torch.tensor( type2ljlk[a] )
            lj_correction_parameters[i,j,0] = (type2hb[a]==HbAtom.DO)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,1] = (type2hb[a]==HbAtom.AC)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,2] = (type2hb[a]==HbAtom.HP)
            lj_correction_parameters[i,j,3] = (a=="SH1" or a=="HS")
    for j,a in enumerate(aa2elt[i]):
        if (a is not None):
            element_index[i,j] = elt2idx[a]

torsion_indices = torch.full((NAATOKENS, NTOTALDOFS, 4), 0)  # 扭转角索引表
torsion_can_flip = torch.full((NAATOKENS, NTOTALDOFS), False, dtype=torch.bool)
for i in range(NPROTAAS):
    i_l, i_a = aa2long[i], aa2longalt[i]

    # protein omega/phi/psi
    torsion_indices[i, 0, :] = torch.tensor([-1, -2, 0, 1])  # omega
    torsion_indices[i, 1, :] = torch.tensor([-2, 0, 1, 2])  # phi
    torsion_indices[i, 2, :] = torch.tensor([0, 1, 2, 3])  # psi (+pi)

    # protein chis
    for j in range(4):
        if torsions[i][j] is None:
            continue
        for k in range(4):
            a = torsions[i][j][k]
            torsion_indices[i, 3 + j, k] = i_l.index(a)
            if (i_l.index(a) != i_a.index(a)):
                torsion_can_flip[i, 3 + j] = True  ##bb tors never flip

    # CB/CG angles (only masking uses these indices)
    torsion_indices[i, 7, :] = torch.tensor([0, 2, 1, 4])  # CB ang1
    torsion_indices[i, 8, :] = torch.tensor([0, 2, 1, 4])  # CB ang2
    torsion_indices[i, 9, :] = torch.tensor([0, 2, 4, 5])  # CG ang (arg 1 ignored)

# 注：下文中的额外氨基酸对应HIS_D的立场计算（cb）参数
# (1) 残基间距
cb_lengths = [[] for i in range(NAATOKENS + 1)]
for cst in cartbonded_data_raw['lengths']:
    res_idx = aa2num[ cst['res'] ]
    cb_lengths[res_idx].append((
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        cst['x0'],cst['K']
    ))
ncst_per_res=max([len(i) for i in cb_lengths])
cb_length_t = torch.zeros(NAATOKENS+1, ncst_per_res,4)
for i in range(NNAPROTAAS+1):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']
    if (len(cb_lengths[src])>0):
        cb_length_t[i,:len(cb_lengths[src]),:] = torch.tensor(cb_lengths[src])
# (2) 角间距
cb_angles = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['angles']:
    res_idx = aa2num[ cst['res'] ]
    cb_angles[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        aa2long[res_idx].index(cst['atm3']),
        cst['x0'],cst['K']
    ) )
ncst_per_res=max([len(i) for i in cb_angles])
cb_angle_t = torch.zeros(NAATOKENS+1,ncst_per_res,5)
for i in range(NNAPROTAAS+1):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']

    if (len(cb_angles[src])>0):
        cb_angle_t[i,:len(cb_angles[src]),:] = torch.tensor(cb_angles[src])
# (3) 内部扭转角
cb_torsions = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['torsions']:
    res_idx = aa2num[ cst['res'] ]
    cb_torsions[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        aa2long[res_idx].index(cst['atm3']),
        aa2long[res_idx].index(cst['atm4']),
        cst['x0'],cst['K'],cst['period']
    ) )
ncst_per_res=max([len(i) for i in cb_torsions])
cb_torsion_t = torch.zeros(NAATOKENS+1,ncst_per_res,7)
cb_torsion_t[...,6]=1.0 # periodicity
for i in range(NNAPROTAAS):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']

    if (len(cb_torsions[src])>0):
        cb_torsion_t[i,:len(cb_torsions[src]),:] = torch.tensor(cb_torsions[src])

# bond graph traversal
num_bonds = torch.zeros((NAATOKENS, NTOTAL, NTOTAL), dtype=torch.long)
for i in range(NNAPROTAAS):
    num_bonds_i = np.zeros((NTOTAL, NTOTAL))
    for (bnamei,bnamej) in aabonds[i]:
        bi,bj = aa2long[i].index(bnamei), aa2long[i].index(bnamej)
        num_bonds_i[bi,bj] = 1
    num_bonds_i = scipy.sparse.csgraph.shortest_path (num_bonds_i,directed=False)
    num_bonds_i[num_bonds_i>=4] = 4
    num_bonds[i,...] = torch.tensor(num_bonds_i)

# 告诉模型异亮氨酸（ILE）的chi_2旋转后是不对称的，甲基的位置是唯一的，不能像苯环或羧基那样进行对称交换
torsion_can_flip[8, 4] = False

"""构建运动学参数"""
# 构建每个原子的基底框架。指定每个刚体/扭转段是绕哪个父原子（或父 frame）旋转的
base_indices = torch.full((NAATOKENS, NTOTAL),0, dtype=torch.long)
# 基准坐标系中每个原子的坐标。
xyzs_in_base_frame = torch.ones((NAATOKENS, NTOTAL, 4))
# 扭转框架
RTs_by_torsion = torch.eye(4).repeat(NAATOKENS, NTOTALTORS, 1, 1)
# 可弯曲角度的参考值
reference_angles = torch.ones((NAATOKENS, NPROTANGS, 2))
# 初始化运动学参数中的蛋白质部分
for i in range(NPROTAAS):
    i_l = aa2long[i]
    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i, idx] = base
        xyzs_in_base_frame[i, idx, :3] = torch.tensor(coords)

    # omega frame
    RTs_by_torsion[i, 0, :3, :3] = torch.eye(3)
    RTs_by_torsion[i, 0, :3, 3] = torch.zeros(3)

    # phi frame
    RTs_by_torsion[i, 1, :3, :3] = make_frame(
        xyzs_in_base_frame[i, 0, :3] - xyzs_in_base_frame[i, 1, :3],
        torch.tensor([1., 0., 0.])
    )
    RTs_by_torsion[i, 1, :3, 3] = xyzs_in_base_frame[i, 0, :3]

    # psi frame
    RTs_by_torsion[i, 2, :3, :3] = make_frame(
        xyzs_in_base_frame[i, 2, :3] - xyzs_in_base_frame[i, 1, :3],
        xyzs_in_base_frame[i, 1, :3] - xyzs_in_base_frame[i, 0, :3]
    )
    RTs_by_torsion[i, 2, :3, 3] = xyzs_in_base_frame[i, 2, :3]

    # chi1 frame
    if torsions[i][0] is not None:
        a0, a1, a2 = torsion_indices[i, 3, 0:3]
        RTs_by_torsion[i, 3, :3, :3] = make_frame(
            xyzs_in_base_frame[i, a2, :3] - xyzs_in_base_frame[i, a1, :3],
            xyzs_in_base_frame[i, a0, :3] - xyzs_in_base_frame[i, a1, :3],
        )
        RTs_by_torsion[i, 3, :3, 3] = xyzs_in_base_frame[i, a2, :3]

    # chi2/3/4 frame
    for j in range(1, 4):
        if torsions[i][j] is not None:
            a2 = torsion_indices[i, 3 + j, 2]
            if ((i == 18 and j == 2) or (i == 8 and j == 2)):  # TYR CZ-OH & HIS CE1-HE1 a special case
                a0, a1 = torsion_indices[i, 3 + j, 0:2]
                RTs_by_torsion[i, 3 + j, :3, :3] = make_frame(
                    xyzs_in_base_frame[i, a2, :3] - xyzs_in_base_frame[i, a1, :3],
                    xyzs_in_base_frame[i, a0, :3] - xyzs_in_base_frame[i, a1, :3])
            else:
                RTs_by_torsion[i, 3 + j, :3, :3] = make_frame(
                    xyzs_in_base_frame[i, a2, :3],
                    torch.tensor([-1., 0., 0.]), )
            RTs_by_torsion[i, 3 + j, :3, 3] = xyzs_in_base_frame[i, a2, :3]

    # CB/CG angles
    NCr = 0.5 * (xyzs_in_base_frame[i, 0, :3] + xyzs_in_base_frame[i, 2, :3])
    CAr = xyzs_in_base_frame[i, 1, :3]
    CBr = xyzs_in_base_frame[i, 4, :3]
    CGr = xyzs_in_base_frame[i, 5, :3]
    reference_angles[i, 0, :] = th_ang_v(CBr - CAr, NCr - CAr)
    NCp = xyzs_in_base_frame[i, 2, :3] - xyzs_in_base_frame[i, 0, :3]
    NCpp = NCp - torch.dot(NCp, NCr) / torch.dot(NCr, NCr) * NCr
    reference_angles[i, 1, :] = th_ang_v(CBr - CAr, NCpp)
    reference_angles[i, 2, :] = th_ang_v(CGr, torch.tensor([-1., 0., 0.]))
# 初始化运动学参数中的小分子部分
xyzs_in_base_frame[NNAPROTAAS:,1, :3] = 0
frame_indices = torch.full((NAATOKENS, NFRAMES, 3, 2),0, dtype=torch.long)
for i in range(NNAPROTAAS):
    i_l = aa2long[i]
    for j, x in enumerate(frames[i]):
        if x is not None:
            # frames are stored as (residue offset, atom position)
            frame_indices[i, j, 0] = torch.tensor((0, i_l.index(x[0])))
            frame_indices[i, j, 1] = torch.tensor((0, i_l.index(x[1])))
            frame_indices[i, j, 2] = torch.tensor((0, i_l.index(x[2])))

if __name__ == '__main__':
    print()
