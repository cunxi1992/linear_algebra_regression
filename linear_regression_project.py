
# coding: utf-8

# 欢迎来到线性回归项目。
# 
# 若项目中的题目有困难没完成也没关系，我们鼓励您带着问题提交项目，评审会给予您诸多帮助。
# 
# 其中证明题可以提交 pdf 格式，手写后扫描或使用公式编辑器（latex，mathtype）均可行。

# # 1 矩阵运算
# 
# ## 1.1 创建一个 4*4 的单位矩阵

# In[96]:


# 这个项目设计来帮你熟悉 python list 和线性代数
# 你不能调用任何python库，包括NumPy，来完成作业

A = [[1,2,3], 
     [2,3,3], 
     [1,2,5]]

B = [[1,2,3,5], 
     [2,3,3,5], 
     [1,2,5,1]]

#TODO 创建一个 4*4 单位矩阵
def identity_matrix(n):
    I = [0] * n
    for i in range(0, n):
        I[i] = [0]*n
        I[i][i] = 1
    return I
print identity_matrix(4)


# ## 1.2 返回矩阵的行数和列数

# In[97]:


# TODO 返回矩阵的行数和列数
def shape(M):
    i = len(M)
    j = len(M[0])
    return i,j


# In[98]:


# 运行以下代码测试你的 shape 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_shape')


# ## 1.3 每个元素四舍五入到特定小数数位

# In[99]:


# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts = 4):
    for x in M:
        for y in range(len(x)):
            x[y] = round(x[y], decPts)
    pass


# In[100]:


# 运行以下代码测试你的 matxRound 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxRound')


# ## 1.4 计算矩阵的转置

# In[101]:


# TODO 计算矩阵的转置
def transpose(M):
    return [list(col) for col in zip(*M)]


# In[102]:


# 运行以下代码测试你的 transpose 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_transpose')


# ## 1.5 计算矩阵乘法 AB

# In[103]:


# TODO 计算矩阵乘法 AB，如果无法相乘则返回None
def matxMultiply(A,B):
    _, c = shape(A)
    r, _ = shape(B)
    if c != r :
        return None

    Bt = transpose(B)
    result = [[sum((a*b) for a,b in zip(row,col)) for col in Bt] for row in A]
    return result


# In[104]:


# 运行以下代码测试你的 matxMultiply 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_matxMultiply')


# ---
# 
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

# In[105]:


# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    return [ra + rb for ra,rb in zip(A,b)]


# In[106]:


# 运行以下代码测试你的 augmentMatrix 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_augmentMatrix')


# ## 2.2 初等行变换
# - 交换两行
# - 把某行乘以一个非零常数
# - 把某行加上另一行的若干倍：

# In[107]:


# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]


# In[108]:


# 运行以下代码测试你的 swapRows 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_swapRows')


# In[109]:


# TODO r1 <--- r1 * scale， scale!=0
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    else:
        M[r] = [i * scale for i in M[r]]


# In[110]:


# 运行以下代码测试你的 scaleRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_scaleRow')


# In[111]:


# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    x = [i * scale for i in M[r2]]
    M[r1] = [a + b for a, b in zip(M[r1], x)]


# In[112]:


# 运行以下代码测试你的 addScaledRow 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_addScaledRow')


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

# In[113]:


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

# 开始定义：
# Gaussian Jordan
def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    #检查A，b是否行数相同
    if len(A) != len(b):
        return None
    else:
        #构造增广矩阵
        Ab = augmentMatrix(A, b)
        #对Ab进行转置，方便遍历元素
        Ab_T = transpose(Ab)
        for index in range(len(Ab_T[0])):
            Ab_T = transpose(Ab)
            max_this_row = abs(Ab_T[index][index])
            max_row_number = index
            for index_2 in range(index, len(Ab_T[0])):
                if abs(Ab_T[index][index_2]) > max_this_row:
                    max_this_row = abs(Ab_T[index][index_2])
                    #记录当前绝对值最大值所在的行号,index_2是转置后该元素的列号，是原矩阵中的行号
                    max_row_number = index_2
            #如果绝对值最大值为0，则A为奇异矩阵
            if max_this_row <= epsilon:
                return None
            else:
                #使用第一个行变换，将绝对值最大值所在行交换到对角线元素所在行（行c）
                swapRows(Ab, max_row_number, index)
                #使用第二个行变换，将列c的对角线元素缩放为1
                scaleRow(Ab, index, 1.0/Ab[index][index])
                #多次使用第三个行变换，将列c的其他元素消为0
                for index_3 in range(len(Ab_T[0])):
                    if index_3 != index:
                        addScaledRow(Ab, index_3, index, (-Ab[index_3][index])/Ab[index][index])
        #返回Ab的最后一列
        Ab_last_col = [[] for i in range(len(Ab_T[0]))]
        for index_4 in range(len(Ab_T[0])):
            Ab_last_col[index_4].append(Ab[index_4][len(Ab_T[0])])
        return Ab_last_col


# In[114]:


# 运行以下代码测试你的 gj_Solve 函数
get_ipython().magic(u'run -i -e test.py LinearRegressionTestCase.test_gj_Solve')


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

# TODO 证明：

# In[115]:


get_ipython().run_cell_magic(u'latex', u'', u'$$\\text{1\uff1a\u8bbe\u65b9\u9635A\u4e3am+n\u884cm+n\u5217\u7684m+n\u9636\u65b9\u9635}$$\n$$\\text{2\uff1a\u6839\u636e\u9898\u76ee\u4e2d\u6240\u7ed9\u6761\u4ef6\u53ef\u77e5\uff0c\u65b9\u9635A\u7684\u5f62\u5f0f\u5982\u4e0b\uff1a}$$\n$$ A = \\begin{bmatrix}\n     &           &              & a_{1, m+1}     & a_{1, m+2}     & ...            & a_{1, m+n}  \\\\\n     &           &              & .              & .              & ...            & .            \\\\\n     & I         &              & .              & .              & ...            & .            \\\\\n     &           &              & .              & .              & ...            & .            \\\\\n     &           &              & a_{m, m+1}     & a_{m, m+2}     & ...            & a_{m, m+n}   \\\\\n0    & ...       & 0            & 0              & a_{m+1, m+2}   & ...            & a_{m+1, m+n} \\\\\n0    & ...       & 0            & 0              & a_{m+2, m+2}   & ...            & a_{m+2,m+n}  \\\\\n.    & ...       & .            & .              & .              & ...            & .             \\\\\n.    & ...       & .            & .              & .              & ...            & .             \\\\\n.    & ...       & .            & .              & .              & ...            & .             \\\\\n0    & ...       & 0            & 0              & a_{m+n, m+2}   & ...            & a_{m+n, m+n}  \\\\\n\\end{bmatrix} $$\n$$\\text{\u5176\u4e2dI\u4e3am\u9636\u5355\u4f4d\u9635\uff0c\u5f62\u5f0f\u5982\u4e0b\uff1a}$$\n$$ I = \\begin{bmatrix}\n1          &0        & ...       & 0      & 0\\\\\n0          &1        & ...       & 0      & 0\\\\\n.          & .       & ...       & .      & .\\\\\n.          & .       & ...       & .      & .\\\\\n.          & .       & ...       & .      & .\\\\\n0          & 0       & ...       & 1      & 0\\\\\n0          & 0       & ...       & 0      & 1\\\\\n\\end{bmatrix} $$\n$$\\text{3\uff1a\u4efb\u53d6\u975e\u96f6m+n\u9636\u65b9\u9635B\u5982\u4e0b\uff1a}$$\n$$ B = \\begin{bmatrix}\n         &           &              & b_{1, m+1}     & b_{1, m+2}     & ...           & b_{1, m+n}  \\\\\n         &           &              & .              & .              & ...            & .            \\\\\n         & J         &              & .              & .              & ...            & .            \\\\\n         &           &              & .              & .              & ...            & .            \\\\\n         &           &              & b_{m, m+1}     & b_{m, m+2}     & ...            & b_{m, m+n}   \\\\\n0        & ...       & 0            & b_{m+1, m+1}   & b_{m+1, m+2}   & ...           & b_{m+1, m+n}  \\\\\n0        & ...       & 0            & 0              & 0              & ...            & 0             \\\\\n.        & ...       & .            & .              & .              & ...            & .             \\\\\n.        & ...       & .            & .              & .              & ...            & .             \\\\\n.        & ...       & .            & .              & .              & ...            & .             \\\\\n0        & ...       & 0            & 0              & 0              & ...            & 0             \\\\\n\\end{bmatrix} $$\n$$\\text{\u5176\u4e2dJ\u4e3am\u9636\u975e\u96f6\u65b9\u9635\uff0c\u6b63\u5bf9\u89d2\u7ebf\u4e0a\u5168\u4e3a0\uff0c\u5176\u4ed6\u5143\u7d20\u5168\u4e3a\u975e0\uff0c\u5f62\u5f0f\u5982\u4e0b\uff1a}$$\n$$ J = \\begin{bmatrix}\n0          & b_{1, 2}        & ...       & b_{1, m-1}      & b_{1, m}\\\\\nb_{2, 1}   &0                & ...       & b_{2, m-1}      & b_{2, m}\\\\\n.          & .               & ...       & .               & .\\\\\n.          & .               & ...       & .               & .\\\\\n.          & .               & ...       & .               & .\\\\\nb_{m-1, 1} & b_{m-1, 2}      & ...       & 0               & b_{m-1, m}\\\\\nb_{m, 1}   & b_{m, 2}        & ...       & b_{m, m-1}      & 0\\\\\n\\end{bmatrix} $$\n$$\\text{4\uff1a\u901a\u8fc7\u8ba1\u7b97\u6211\u4eec\u53ef\u4ee5\u77e5\u9053\uff0cBA = 0\uff0c\u6839\u636e\u77e9\u9635\u6027\u8d28\u53ef\u5f97\uff1ar(B) + r(A) \u2264 n\uff0c\u6240\u4ee5r(A) \u2264 n - r(B)}$$\n$$\\text{5\uff1a\u56e0\u4e3a\u65b9\u9635B\u662f\u975e\u96f6m+n\u9636\u65b9\u9635\uff0c\u4e14\u65b9\u9635B\u4ece\u7b2cm+2\u884c\u5230\u7b2cm+n\u884c\u6240\u6709\u5143\u7d20\u4e3a0\uff0c\u6240\u4ee50 \uff1c r(B) \uff1c n}$$\n$$\\text{6\uff1a\u5047\u8bber(A) = n\uff0c\u5219r(B) = 0\uff0c\u4e0e\u6b65\u9aa45\u4e2d0 \uff1c r(B) \uff1c n\u77db\u76fe\uff0c\u6240\u4ee5r(A) \uff1c n}$$\n$$\\text{7\uff1a\u6240\u4ee5n\u9636\u65b9\u9635A\u4e3a\u5947\u5f02\u77e9\u9635}$$')


# ---
# 
# # 3 线性回归: 
# 
# ## 3.1 计算损失函数相对于参数的导数 (两个3.1 选做其一)
# 
# 我们定义损失函数 $E$ ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 定义向量$Y$, 矩阵$X$ 和向量$h$ :
# $$
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

# TODO 证明：

# In[116]:


get_ipython().run_cell_magic(u'latex', u'', u'\n$$\\\\$$\n\n$$\n\\text{\u53d8\u6362\u540e\u7684 }X^T\\text{\u4e3a\uff1a }\n$$\n\n$$\nX^T =  \\begin{bmatrix}\nx_1 & x_2 & ... & x_n\\\\\n1 & 1 & ... & 1\\\\\n\\end{bmatrix}\n$$\n\n$$\\\\$$\n\n$$ \n\\text{\u5df2\u77e5\uff1a }\nY =  \\begin{bmatrix}\ny_1 \\\\\ny_2 \\\\\n... \\\\\ny_n\n\\end{bmatrix}\n,\nX =  \\begin{bmatrix}\nx_1 & 1 \\\\\nx_2 & 1\\\\\n... & ...\\\\\nx_n & 1 \\\\\n\\end{bmatrix},\nh =  \\begin{bmatrix}\nm \\\\\nb \\\\\n\\end{bmatrix}\n$$\n\n$$\n\\text{\u63a8\u5bfc\u51fa }\\text{\uff1a }\n$$\n\n$$\n2X^TXh - 2X^TY = -2X^T(Y-Xh)\n$$\n\n\n$$\\\\$$\n\n$$\nY-Xh = \\begin{bmatrix}\ny_1 - mx_1 - b \\\\\ny_2 - mx_2 - b \\\\\n... \\\\\ny_n - mx_n - b \\\\\n\\end{bmatrix}\n$$\n\n\n\n\n$$\n\\text{\u6240\u4ee5 }\\text{\uff1a }\n$$\n\n$$\n2X^TXh - 2X^TY = -2X^T\\begin{bmatrix}\ny_1 - mx_1 - b \\\\\ny_2 - mx_2 - b \\\\\n... \\\\\ny_n - mx_n - b \\\\\n\\end{bmatrix} = \\begin{bmatrix}\n\\sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)} \\\\\n\\sum_{i=1}^{n}{-2(y_i - mx_i - b)}\n\\end{bmatrix}\n$$\n\n$$\n\\text{\u7136\u540e\u5bf9\u5b9a\u4e49\u7684\u635f\u5931\u51fd\u6570(E)\u6c42m\u3001b\u7684\u504f\u5bfc }\\text{\uff1a }\n$$\n\n\n$$\n\\frac{\\partial E}{\\partial m} = \\sum_{i=1}^{n}{2(y_i - mx_i - b)\\frac{\\partial {(y_i - mx_i - b)}}{\\partial m}} $$\n$$= \\sum_{i=1}^{n}{2(y_i - mx_i - b)(-x_i)} $$\n$$= \\sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\n$$\n\n\n$$\n\\frac{\\partial E}{\\partial b} = \\sum_{i=1}^{n}{2(y_i - mx_i - b) \\frac{\\partial {(y_i - mx_i - b)}}{\\partial b}} $$\n$$= \\sum_{i=1}^{n}{2(y_i - mx_i - b)(-1)} $$\n$$= \\sum_{i=1}^{n}{-2(y_i - mx_i - b)}\n$$\n\n\n$$\\\\$$\n$$\\\\$$\n\n$$\n\\text{\u7efc\u4e0a\u6240\u8ff0 }\\text{\uff1a }\n$$\n\n\n$$\n\\frac{\\partial E}{\\partial m} = \\sum_{i=1}^{n}{-2x_i(y_i - mx_i - b)}\n$$\n\n$$\n\\frac{\\partial E}{\\partial b} = \\sum_{i=1}^{n}{-2(y_i - mx_i - b)}\n$$\n\n\n\n$$\n\\begin{bmatrix}\n\\frac{\\partial E}{\\partial m} \\\\\n\\frac{\\partial E}{\\partial b} \n\\end{bmatrix} = 2X^TXh - 2X^TY\n$$')


# ## 3.1 计算损失函数相对于参数的导数（两个3.1 选做其一）
# 
# 我们定义损失函数 $E$ ：
# $$
# E = \sum_{i=1}^{n}{(y_i - mx_i - b)^2}
# $$
# 定义向量$Y$, 矩阵$X$ 和向量$h$ :
# $$
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

# TODO 请使用 latex （请参照题目的 latex 写法学习）
# 
# TODO 证明：

# ## 3.2  线性回归
# 
# ### 求解方程 $X^TXh = X^TY $, 计算线性回归的最佳参数 h 
# *如果你想更深入地了解Normal Equation是如何做线性回归的，可以看看MIT的线性代数公开课，相关内容在[投影矩阵与最小二乘](http://open.163.com/movie/2010/11/P/U/M6V0BQC4M_M6V2AOJPU.html)。*

# In[117]:


# TODO 实现线性回归
'''
参数：(x,y) 二元组列表
返回：m，b
'''
#线性回归方程
def linearRegression(points):
    # 构建 Ax = b 的线性方程
    X = [[points[i][0], 1] for i in range(len(points))]
   
    Y = [[points[i][1]] for i in range(len(points))]

    X_T = transpose(X)
  
    A = matxMultiply(X_T, X)
    
    b = matxMultiply(X_T, Y)
   

    #m, b = (i[0] for i in gj_Solve(A, b))
    m, b = (i[0] for i in gj_Solve(A, b, decPts=4, epsilon=1.0e-16))
    return m, b


# ## 3.3 测试你的线性回归实现

# In[118]:


# TODO 构造线性函数

# 构造线性函数

m = 2
b = 3

# TODO 构造 100 个线性函数上的点，加上适当的高斯噪音

print '构造100个线性函数上的点，加上适当的高斯噪音，均值为0方差为1'

import random

x_origin = [random.uniform(-50,50) for i in range(100)]
Y_origin = [i * m + b for i in x_origin]

x_gauss = [random.gauss(0, 1) for i in range(100)]
y_gauss = [random.gauss(0, 1) for i in range(100)]

xx = [x + y for x,y in zip(x_origin, x_gauss)]
yy = [x + y for x,y in zip(Y_origin, y_gauss)]

points = [(x,y) for x,y in zip(xx, yy)]

#TODO 对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较

print '对这100个点进行线性回归，将线性回归得到的函数和原线性函数比较'

m_gauss, b_gauss = linearRegression(points)

print '原始m, b '
print m, b

print '处理后的m, b'
print m_gauss, b_gauss

print '备注：此为x和y都增加高斯噪音的数据，根据需要可以不加x的高斯噪音'

