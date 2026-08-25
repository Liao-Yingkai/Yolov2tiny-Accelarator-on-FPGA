设计原则：

- 输入：NHWC格式 feature map stream
- 内部：Line Buffer + Window Generator
- 支持：
  - kernel=3×3（当前项目固定）
  - padding=1
  - stride=1（第一版固定）
  - P路空间并行窗口
  - 每cycle产生P个patch对应的第k维数据
- 输出直接连接systolic

---

# Streaming Im2Row Module Design Specification

## 1. 模块功能

### 模块名称

```
im2row_window_generator
```

### 功能

将输入feature map：

\[
(N,H,W,C_{in})
\]


转换为适合矩阵乘法加速器的数据流：

\[
(H_oW_o)\times(K_hK_wC_{in})
\]


但是不显式生成完整im2row矩阵，而是：

- 使用line buffer缓存空间邻域
- 使用window register形成卷积窗口
- 按照systolic需求输出activation vector


数据流：

```
NHWC Feature Stream

        |
        v

Line Buffer

        |
        v

3x3 Window Generator

        |
        v

Flatten

        |
        v

P-way Activation Stream

        |
        v

Systolic Array
```

---

# 2. 设计目标

假设：

## 输入

```
NHWC
```

例如：

Conv2：

```
H = 14
W = 14
Cin = 8
```


每个空间位置：

```
pixel = [c0,c1,...c(Cin-1)]
```


---

## Kernel

当前固定：

```
Kh = 3
Kw = 3
```


展开维度：

\[
K=Kh\times Kw\times Cin
\]


例如：

Conv2:

\[
K=3\times3\times8=72
\]


---

## 并行度

定义：

```
P = number of parallel windows
```


例如：

```
P=4
```


表示：

每cycle生成：

```
4个空间位置
```

对应：

矩阵：

\[
A_{tile}=P\times K
\]


---

# 3. 顶层接口设计

## 输入接口


### Feature map stream

```cpp
hls::stream<pixel_t> input_stream
```


其中：

```cpp
struct pixel_t
{
    ap_int<8> data[CIN];
};
```


含义：

一次输入：

一个HW位置的所有channel。


例如：

Conv2:

```
pixel_t

{
 c0
 c1
 ...
 c7
}
```


---

## 控制信号

### 输入尺寸

```cpp
int H;
int W;
```


作用：

用于：

- 行结束判断
- padding判断
- 地址管理


---

### Kernel参数

第一版固定：

```cpp
KH = 3;
KW = 3;
```

无需输入。


---

### Padding

第一版固定：

```
padding = 1
```

无需输入。


---

### stride

第一版固定：

```
stride = 1
```

无需输入。


---

# 4. 输出接口


## 输出给systolic

不要输出完整patch。


即：

不要：

```
P × (KhKwCin)
```


原因：

一次传输过宽。


---

采用：

K维stream。


定义：

```cpp
struct activation_pack
{
    ap_int<8> data[P];
};
```


含义：

一个cycle：

输出：

```
P个空间位置
```

的同一个k维元素。


例如：

P=4:


cycle k=0:


```
data[0]=patch0[k0]

data[1]=patch1[k0]

data[2]=patch2[k0]

data[3]=patch3[k0]
```


---

输出：

```cpp
hls::stream<activation_pack> output_stream
```


---

# 5. 内部模块结构


整体：

```
              input_stream

                    |
                    v

        +-----------------------+
        |   Pixel Counter       |
        |                       |
        | row_counter           |
        | col_counter           |
        +-----------------------+

                    |
                    v


        +-----------------------+
        |    Line Buffer        |
        +-----------------------+

                    |
                    v


        +-----------------------+
        | Window Generator      |
        |                       |
        | 3x3xCin windows       |
        +-----------------------+

                    |
                    v


        +-----------------------+
        | Flatten & Scheduler   |
        +-----------------------+

                    |
                    v

             output_stream

```

---

# 6. Pixel Counter设计


内部维护：

```cpp
int row;
int col;
```


每收到一个pixel：

执行：

```cpp
if(col == W-1)
{
    col = 0;
    row++;
}
else
{
    col++;
}
```


作用：

知道：

- 当前pixel坐标
- 是否换行
- 是否padding


---

# 7. Line Buffer设计


对于：

3×3 kernel


需要：

\[
Kh-1=2
\]


行buffer。


结构：

```cpp
pixel_t line_buffer[2][MAX_W];
```


例如：

```
line_buffer[0]

上一上一行


line_buffer[1]

上一行
```


当前输入：

```
current pixel
```


共同组成：

三行数据。


---

## 更新逻辑


每输入一个pixel：

发生shift：

```
line_buffer[0]
← line_buffer[1]

line_buffer[1]
← current row
```

实际实现：

采用循环buffer避免大量copy。


---

# 8. Window Register设计


维护：

3×3窗口。


注意：

每个元素是：

pixel vector。


即：

```cpp
pixel_t window[3][3];
```


展开：

```
window[row][col].channel
```


---

## 每输入一个pixel


执行：

### Step 1

右侧注入：

```
window[0][2]

=
line_buffer_row0


window[1][2]

=
line_buffer_row1


window[2][2]

=
current_pixel
```


---

### Step 2

窗口左移


例如：

之前：

```
A B C

D E F

G H I
```


之后：

```
B C new

E F new

H I new
```


对应：

```cpp
window[x][0]
=
window[x][1]

window[x][1]
=
window[x][2]
```


---

# 9. Padding处理


不创建padding feature map。


采用在线判断。


当window需要访问：

```
row<0
row>=H

col<0
col>=W
```


直接产生：

```cpp
pixel_t zero_pixel;
```

例如：

边界：

```
0 0 0

0 x x

0 x x
```


其中0：

由逻辑产生。


---

# 10. P窗口并行设计


目标：

每cycle产生：

P个window。


例如：

P=4:


输入：

一次推进：

```
pixel0
pixel1
pixel2
pixel3
```


产生：

```
window0

window1

window2

window3
```


---

结构：

```
              Line Buffer

                    |

        +-----------+-----------+
        |           |           |

     Window0     Window1     Window2 ... WindowP


                    |

              Flatten


                    |

          activation_pack

```

---

# 11. Flatten顺序


必须和weight layout一致。


采用：

```
kh -> kw -> cin
```


即：

index：

\[
idx=(kh\times Kw+kw)\times Cin+cin
\]


例如：

Cin=8:

```
k0:

kh0 kw0 c0

k1:

kh0 kw0 c1

...

k8:

kh0 kw1 c0

```

---

对应weight：

```
weight[K][Cout]
```

必须保持同样顺序。

---

# 12. 输出Scheduler


窗口形成以后：

不能一次输出：

```
P × K
```


而是：

沿K维扫描。


内部：

维护：

```cpp
k_counter;
```


范围：

```
0 ~ Kh*Kw*Cin-1
```


每cycle：

输出：

```cpp
activation_pack
```


例如：

k=0:

```
patch0[0]
patch1[0]
patch2[0]
patch3[0]
```


k=1:

```
patch0[1]
...
```


直到：

```
k=K-1
```


---

# 13. HLS Interface建议


顶层：

```cpp
void im2row(
    hls::stream<pixel_t>& input,
    hls::stream<activation_pack>& output,
    int H,
    int W
)
```


---

接口：

```cpp
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output

#pragma HLS INTERFACE s_axilite port=H
#pragma HLS INTERFACE s_axilite port=W
#pragma HLS INTERFACE s_axilite port=return
```


---

# 14. HLS优化要求


## DATAFLOW

模块：

```cpp
#pragma HLS DATAFLOW
```


划分：

```
read_input

line_buffer

window_generator

output_scheduler

```


---

## ARRAY_PARTITION


Channel并行：

```cpp
#pragma HLS ARRAY_PARTITION variable=pixel complete dim=1
```


---

Window：

```cpp
#pragma HLS ARRAY_PARTITION variable=window complete
```


---

# 15. 性能目标


目标：

II=1。


即：

稳定情况下：

每cycle：

输出：

```
P activation
```


供给：

systolic：

```
P activation/cycle
```


---

# 16. 与Systolic接口对应关系


Im2row输出：

```
activation_pack

[P]
```


Weight模块输出：

```
weight_pack

[P]
```


每cycle：

PE阵列：

执行：

\[
P\times P
\]


MAC。


---

# 17. 第一版实现建议

固定参数：

```
Cin <= 8
Kernel=3x3
stride=1
padding=1
P=4
```


不要立即参数化所有情况。

先验证：

1. 单窗口正确性
2. P窗口正确性
3. padding正确性
4. stream吞吐
5. 与systolic联调


---

# 最终模块数据流总结

```
NHWC pixel stream

        |
        v

row/col counter

        |
        v

2-row line buffer

        |
        v

3x3 sliding window

        |
        v

P parallel windows

        |
        v

flatten:
kh -> kw -> cin

        |
        v

k dimension scheduler

        |
        v

activation_pack[P]

        |
        v

P×P systolic array

```