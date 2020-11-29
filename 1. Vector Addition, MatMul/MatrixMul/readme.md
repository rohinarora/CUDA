#### Usage

```
nvcc -arch=sm_35 -O3 VectorMatMul.cu  -o VectorMatMul && ./VectorMatMul 1000 1000
./VectorMatMul <rows> <columns>
```

* 20x speed up for 1000x1000 matrix compared to CPU. Block size 256 (16x16)