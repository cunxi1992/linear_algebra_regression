
# coding: utf-8

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[1]:

# 这个项目设计来帮你熟悉 python range 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
I =  [ [ 0 for i in range(4)] for j in range(4) ]
for i in range(4):
    for j in range(4):
        if i==j:
            I[i][j]=1


# ## 1.2 返回矩阵的行数和列数

# In[2]:

# TODO 返回矩阵的行数和列数
def shape(M):
    row = len(M)
    column = len(M[0])
    return row,column


# ## 1.3 每个元素四舍五入到特定小数数位

# In[3]:

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(shape(M)[0]):
        for j in range(shape(M)[1]):
            M[i][j]=round(M[i][j],decPts)


# ## 1.4 计算矩阵的转置

# In[4]:

# TODO 计算矩阵的转置
def transpose(M):
# 方法1.使用zip函数
# 先用zip函数得到“行列互换”的效果，
# 再通过map函数对每个元素应用list()函数，将里面的tuple转换为list

# Python 内置的高阶函数map(func, seq1[, seq2,…])
# map()函数将func作用于seq中的每一个元素，
# 并将所有的调用的结果作为一个list返回。
# 此例map()接收一个函数list（）和一个 list，
# 过把list（）依次作用在list的每个元素上，得到一个新的list并返回。
    W = map(list,zip(*M))
    
# 或者用列表推导式如下表达:
# 通过列表推导式（list comprehensions），对每个元素应用list()函数，将里面的tuple转换为list
#    return [list(col) for col in zip(*M)]
    
# 方法2.使用numpy库
#   W=M.T 

# 方法3.参考
# 多次调用shape函数，导致效率不高
#    W =  [ [ 0 for i in range(shape(M)[0]) ] for j in range(shape(M)[1]) ]
#    for i in range(shape(M)[0]):
#        for j in range(shape(M)[1]):
#            W[j][i]=M[i][j]
    return W

print transpose(A)


# ### zip()函数的理解

# In[5]:

def ziptest(M):
#zip函数接受任意多个（包括0个和1个）序列作为参数，返回的是一个tuple（元组），然后返回由这些tuples组成的list（列表）
    return zip(M)

print ziptest(A)


# In[6]:

def transpose(M):
# *M表示将M列表中的元素来zip，当M列表中存的是列表元素时，就是将列表元素来zip
# zip(*M)这里相当于zip([1,2,3],[4,5,6],[7,8,9])
    return zip(*M)
    
print transpose(A)


# ### 比较实现之间的性能差距

# In[7]:

import numpy as np
import profile

# def transpose_student_1(M):
#     # 多次调用shape函数，导致效率不高
#     W =  [ [ 0 for i in range(shape(M)[0]) ] for j in range(shape(M)[1]) ]
#     for i in range(shape(M)[0]):
#         for j in range(shape(M)[1]):
#             W[j][i]=M[i][j]

def transpose_student_2(M):
    W = map(list,zip(*M))

def transpose_pythonic(M):
    return [list(col) for col in zip(*M)]

def test(times=10000):
    for t in range(times):
        r,c = np.random.randint(5,25,2)
        matrix = np.random.randint(-10,11,size=(r,c)).tolist()

        transpose_student_2(matrix)        
        transpose_pythonic(matrix)

if __name__ == '__main__':
    # 你可以修改测试的次数以获得更明显的耗时对比
    profile.run("test(times=10000)")


# 对比；
# 
#  transpose_student_2耗时
#  
#  transpose_pythonic耗时
#  
#  最后一行：总耗时

# ## 1.5 计算矩阵乘法 AB

# In[8]:

# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
# def matxMultiply(A, B):
#     W =  [ [ 0 for i in range(shape(B)[1]) ] for j in range(shape(A)[0])]
#     if shape(A)[1]==shape(B)[0]:
#         k=shape(A)[1]
#         for ai in range(shape(A)[0]):
#             for bj in range(shape(B)[1]):
#                 for kk in range(k):
#                     W[ai][bj]+=A[ai][kk]*B[kk][bj]
#         return W
#     else:
#         return None

#循环写太深会造成代码的可读性变差，你可以考虑利用sum函数取代最内层的循环。
#https://python3-cookbook.readthedocs.io/zh_CN/latest/c01/p19_transform_and_reduce_data_same_time.html
#可能会在改写的过程中发现难以（优雅地）获得矩阵B的列，这里的小诀窍是先将矩阵B转置。
#如果还想进一步压缩代码量，可以将剩下的两层循环用列表推导式替代。
#推荐
def matxMultiply(A,B):
    _, c = shape(A)
    r, _ = shape(B)
    if c != r :
        return None
    Bt = transpose(B)
    result = [[sum((a*b) for a,b in zip(row,col)) for col in Bt] for row in A]
    return result

print "A矩阵与B矩阵相乘为:" 
print matxMultiply(A,B)


# ## 1.6 测试你的函数是否实现正确

# **提示：** 你可以用`from pprint import pprint`来更漂亮的打印数据，详见[用法示例](http://cn-static.udacity.com/mlnd/images/pprint.png)和[文档说明](https://docs.python.org/2/library/pprint.html#pprint.pprint)。

# In[9]:

import pprint
pp=pprint.PrettyPrinter(indent=1,width=40)

#assert断言用法： 如果代码是正确的，那么就不会产生任何结果；反之，Python会抛出断言错误

#TODO 测试1.2 返回矩阵的行和列
M=[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
assert shape(M) == (4, 3)
pp.pprint(shape(M))

#TODO 测试1.3 每个元素四舍五入到特定小数数位
M=[[6.123456,5.123456],[4.123456,3.123456],[2.123456,1.123456]]
matxRound(M,2)
assert M == [[6.12, 5.12],[4.12, 3.12],[2.12, 1.12]]
pp.pprint(M)

#TODO 测试1.4 计算矩阵的转置
M=[[1,4,3,2],[4,6,9,3]]
assert transpose(M) == [(1, 4), (4, 6), (3, 9), (2, 3)]
pp.pprint(transpose(M))

#TODO 测试1.5 计算矩阵乘法AB，AB可以相乘
A=[[1,2],[3,4],[5,6],[7,8],[9,2]]
B=[[1,2,3,4],[4,5,6,7]]
assert matxMultiply(A,B)==[[9, 12, 15, 18], [19, 26, 33, 40],[29, 40, 51, 62],[39, 54, 69, 84],[17, 28, 39, 50]]
pp.pprint(matxMultiply(A,B))

#TODO 测试1.5 计算矩阵乘法AB，AB无法相乘
A=[[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
B=[[1,2,3,4],[1,2,3,4]]
pp.pprint(matxMultiply(A,B))


# # 2 Gaussign Jordan 消元法
# 
# ## 2.1 构造增广矩阵
# 
# $ A = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n}\\
#     a_{21}    & a_{22} & ... & a_{2n}\\
#     a_{31}    & a_{22} & ... & a_{3n}\\
#     ...    & ... & ... & ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn}\\
# \end{bmatrix} , b = \begin{bmatrix}
#     b_{1}  \\
#     b_{2}  \\
#     b_{3}  \\
#     ...    \\
#     b_{n}  \\
# \end{bmatrix}$
# 
# 返回 $ Ab = \begin{bmatrix}
#     a_{11}    & a_{12} & ... & a_{1n} & b_{1}\\
#     a_{21}    & a_{22} & ... & a_{2n} & b_{2}\\
#     a_{31}    & a_{22} & ... & a_{3n} & b_{3}\\
#     ...    & ... & ... & ...& ...\\
#     a_{n1}    & a_{n2} & ... & a_{nn} & b_{n} \end{bmatrix}$

# In[10]:

import copy

# TODO 构造增广矩阵，假设A，b行数相同
# def augmentMatrix(A, b):
#     for i in range(shape(A)[0]):
#         A[i].append(b[i][0])
#     return A

def augmentMatrix(A, b):
    B = copy.deepcopy(A)
    for i in range(shape(B)[0]):
        B[i].append(b[i][0])
    return B

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

b=[[7],[8],[9]]

pp.pprint(augmentMatrix(A,b))


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[11]:

# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1],M[r2]=M[r2],M[r1]

# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale==0:
        raise ValueError('scale could not be zero!')
    else:
        for i in range(shape(M)[1]):
            M[r][i]*=scale

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    r=[ 0 for i in range(shape(M)[1]) ]
    for i in range(shape(M)[1]):
        r[i]=M[r2][i]*scale
    for i in range(shape(M)[1]):
        M[r1][i]+=r[i]


# ## 2.3  Gaussian Jordan 消元法求解 Ax = b

# ### 提示：
# 
# 步骤1 检查A，b是否行数相同
# 
# 步骤2 构造增广矩阵Ab
# 
# 步骤3 逐列转换Ab为化简行阶梯形矩阵 [中文维基链接](https://zh.wikipedia.org/wiki/%E9%98%B6%E6%A2%AF%E5%BD%A2%E7%9F%A9%E9%98%B5#.E5.8C.96.E7.AE.80.E5.90.8E.E7.9A.84-.7Bzh-hans:.E8.A1.8C.3B_zh-hant:.E5.88.97.3B.7D-.E9.98.B6.E6.A2.AF.E5.BD.A2.E7.9F.A9.E9.98.B5)
#     
#     对于Ab的每一列（最后一列除外）
#         当前列为列c
#         寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
#         如果绝对值最大值为0
#             那么A为奇异矩阵，返回None （请在问题2.4中证明该命题）
#         否则
#             使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c） 
#             使用第二个行变换，将列c的对角线元素缩放为1
#             多次使用第三个行变换，将列c的其他元素消为0
#             
# 步骤4 返回Ab的最后一列
# 
# ### 注：
# 我们并没有按照常规方法先把矩阵转化为行阶梯形矩阵，再转换为化简行阶梯形矩阵，而是一步到位。如果你熟悉常规方法的话，可以思考一下两者的等价性。

# In[12]:

# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""

def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    if shape(A)[0]==len(b):

#       转为增广矩阵
        A = augmentMatrix(A,b)

#       针对每列做操作
        for j in range(shape(A)[1]-1):
            i=j
            
            # 在这一步不应该交换行，交换行是需要消耗时间的，这种实现会非常慢
#            while i+1<shape(A)[0]:
#                if A[j][j]<A[i+1][j]:
#                    swapRows(A, j, i+1)
#                i+=1

#           如果为奇异矩阵返回none
            if abs(A[j][j])<epsilon:
                return None
            scaleRow(A,j,1.0/A[j][j])
            for i in range(shape(A)[0]):
                if i!=j:
                    addScaledRow(A, i, j, -1.0*A[i][j])
        matxRound(A,decPts)
        N=transpose(A)[-1]
        return [[N[j]] for j in range(len(N))]
    else:
        raise ValueError


# ## 2.4 证明下面的命题：
# 
# **如果方阵 A 可以被分为4个部分: ** 
# 
# $ A = \begin{bmatrix}
#     I    & X \\
#     Z    & Y \\
# \end{bmatrix} , \text{其中 I 为单位矩阵，Z 为全0矩阵，Y 的第一列全0}$，
# 
# **那么A为奇异矩阵。**
# 
# 提示：从多种角度都可以完成证明
# - 考虑矩阵 Y 和 矩阵 A 的秩
# - 考虑矩阵 Y 和 矩阵 A 的行列式
# - 考虑矩阵 A 的某一列是其他列的线性组合

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：
# 
# **奇异矩阵在线性代数的概念： 对应的行列式等于0的方阵。**
# 
# **在此例中，因为I为单位矩阵,Z为全0矩阵,且A为方阵,由于Y的第一列全0代表矩阵A的对角线含有0,**
# 
# $ A = \begin{bmatrix}
#     1&0&0    & X&X&X \\
#     0&1&0    & X&X&X \\
#     0&0&1    & X&X&X \\   
#     0&0&0    & 0&X&X \\
#     0&0&0    & 0&X&X \\
#     0&0&0    & 0&X&X \\  
# \end{bmatrix} 
# $
# 
# **所以此矩阵转换为上三角形矩阵的行列式的值为0，即|A|=0,**
# 
# **当一个矩阵所在的行列式的值为0的话,该矩阵为奇异矩阵。**
# **因此A为奇异矩阵。**

# ## 2.5 测试 gj_Solve() 实现是否正确

# In[13]:

# TODO 构造 矩阵A，列向量b，其中 A 为奇异矩阵
A=[[3,6],[0,0]]
b=[[0],[3]]
print "此Ax = b的解为:"
pp.pprint(gj_Solve(A,b,2))

# TODO 构造 矩阵A，列向量b，其中 A 为非奇异矩阵
A=[[3,6],[2,1]]
b=[[0],[3]]
# TODO 求解 x 使得 Ax = b
print "此Ax = b的解为:"
pp.pprint(gj_Solve(A,b,2))

# TODO 计算 Ax
A1=3*2-6
A2=2*2-1
Ax=[[0],[3]]

# TODO 比较 Ax 与 b
print "其中，Ax的值为:"
print Ax
print "其中，b的值为:"
print b


# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 E ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 
# 证明：
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：
# 
# 1:
# $$
# E = \sum_{i=1}^{n}{y_i^2-y_ix_im-by_i-y_ix_im+x_i^2m^2+bx_im-by_i+bx_im+b^2}
# $$
# 方程两边对于m同时求导
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_iy_i+2mx_i^2+2bx_i}
# $$
# $$
# \frac{\partial E}{\partial m} = \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}
# $$
# 2:
# $$
# E = \sum_{i=1}^{n}{y_i^2-y_ix_im-by_i-y_ix_im+x_i^2m^2+bx_im-by_i+bx_im+b^2}
# $$
# 方程两边对于b同时求导
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2y_i+2mx_i+2b}
# $$
# $$
# \frac{\partial E}{\partial b} = \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# $$
# 3:
# 将Y和X的值代入
# $$
# 2X^TXh - 2X^TY=2 \begin{bmatrix}
#     m(x_1^2+x_2^2+...+x_n^2)+b(x_1+x_2+...+x_n)\\
#     m(x_1+x_2+...+x_n)+bn 
# \end{bmatrix}-2 \begin{bmatrix}
#     x_1y_1+x_2y_2+...+x_ny_n\\
#     y_1+y_2+...+y_n
# \end{bmatrix}
# $$
# $$
# 2X^TXh - 2X^TY=2 \begin{bmatrix}
#     m(x_1^2+x_2^2+...+x_n^2)+b(x_1+x_2+...+x_n)-(x_1y_1+x_2y_2+...+x_ny_n)\\
#     m(x_1+x_2+...+x_n)+bn-(y_1+y_2+...+y_n)
# \end{bmatrix}
# $$
# $$
# 2X^TXh - 2X^TY= \begin{bmatrix}
# \sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\\
# \sum_{i=1}^{n}{-2(y_i - mx_i - b)}
# \end{bmatrix}
# $$
# 根据1和2的结论可得:
# $$
# 2X^TXh - 2X^TY=
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix} 
# $$

# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 证明：
# 
# $$
# E = Y^TY -2(Xh)^TY + (Xh)^TXh
# $$ 
# 
# $$
# \begin{bmatrix}
#     \frac{\partial E}{\partial m} \\
#     \frac{\partial E}{\partial b} 
# \end{bmatrix}  = \frac{\partial E}{\partial h} = 2X^TXh - 2X^TY
# $$
# 
# $$ 
# \text{其中 }
# Y =  \begin{bmatrix}
#     y_1 \\
#     y_2 \\
#     ... \\
#     y_n
# \end{bmatrix}
# ,
# X =  \begin{bmatrix}
#     x_1 & 1 \\
#     x_2 & 1\\
#     ... & ...\\
#     x_n & 1 \\
# \end{bmatrix},
# h =  \begin{bmatrix}
#     m \\
#     b \\
# \end{bmatrix}
# $$

# TODO 请使用 latex （参照题目的 latex写法学习）
# 
# TODO 证明：
# 
# 

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h

# In[14]:

# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
def linearRegression(points):
    n=len(points)
    x=[[points[i][0],1] for i in range(n)]
    x_t=transpose(x)
    y=[[points[i][1]] for i in range(n)]
    x_t_change_x=matxMultiply(x_t,x)
    x_t_change_y=matxMultiply(x_t,y)
    #print 'x_t_x=',x_t_change_x
    return gj_Solve(x_t_change_x,x_t_change_y)


# ## 3.3 测试你的线性回归实现

# In[15]:

# TODO 构造线性函数

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音
import random
#构建线性函数y=10+2x
P =  [ [ 0 for i in range(2) ] for j in range(100) ]
for i in range(100):
    P[i][0]=random.randint(0, 100)
    P[i][1]=random.gauss(2*P[i][0]+10,1)
#高斯噪音指的是服从标准正态分布的随机数，要求均值为0方差为1
    
#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较
print linearRegression(P)


# ## 4.1 单元测试

# 请确保你的实现通过了以下所有单元测试。

# In[16]:

import unittest
import numpy as np

from decimal import *

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c))


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'))


    def test_transpose(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r))
            self.assertTrue((matrix.T == t).all())


    def test_matxMultiply(self):

        for _ in range(10):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1,mat2))

            self.assertTrue((dotProduct == dp).all())


    def test_augmentMatrix(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))

            Ab = np.array(augmentMatrix(A.tolist(),b.tolist()))
            ab = np.hstack((A,b))

            self.assertTrue((Ab == ab).all())

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all())

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all())
    
    def test_addScaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all())


    def test_gj_Solve(self):

        for _ in range(10):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))
            x = gj_Solve(A.tolist(),b.tolist())
            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None)
            else:
                # Ax = matxMultiply(A.tolist(),x)
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                # print Ax
                # print loss
                self.assertTrue(loss<0.1)


suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
unittest.TextTestRunner(verbosity=3).run(suite)

