* 20x speed up for 1000x1000 matrix
* nvcc -arch=sm_35 -O3 VectorMatMul.cu  -o VectorMatMul && ./VectorMatMul 1000 1000
