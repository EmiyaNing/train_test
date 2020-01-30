# 实现train时的教训
1. 运算部分还是采用armadillo库来进行运算会比较舒服
2. opencv还是只能用来进行图片的读写

## 完整的代码所具备的部件
1. sigmod 运算函数
2. 前传后传函数 propagate
3. 优化函数  optimize
4. 预测函数 predict
5. 模型函数 model

## sigmod函数
参数: float z
功能：计算result = 1/(1+e^(-z))
返回：返回result

## 前传函数propagate
参数：vector<float> w, float b , Mat<float> X(4096, 20000), vector<float> Y
功能: 计算 for(int i = 0; i < 20000; i++){A[i] = sigmod(b + w*X[i])； log_alpha[i] = log(A[i]); log_beta[i] = log(1 - A[i]); temp[i] = (1 - Y[i]);}
     计算 cost = -(Y*log_alpha + log_beta * temp)/20000;
     计算 dw, 计算X的每一列与向量(A-Y)的点积，得到的所有向量的每个元素都除去20000.。。
     计算 db, 计算向量(A-Y)的所有元素的和，并求和取平均数。


## 优化函数optimize
参数：vector<float> w, float b , Mat<float> X(4096, 20000), vector<float> Y, int num_iterations, float learning_rate, bool print_cost
功能：循环num_iterations 次，每次循环调用一次propagate函数，调用完成之后更新w和b，顺带可以打印参数。。

## 预测函数predict
参数：vector<float> w, float b , vector<float> x
功能： 预测x是否为猫。。

## 模型函数model
功能： 整合上述函数的功能。。


# 现在的问题：
1. 矩阵运算部分的矩阵X怎么构造有点麻烦。。