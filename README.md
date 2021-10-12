
# ConvMixer

ConvMixer 非官方实现

pytorch 版本已经实现。

nets 是重构版本 ，test 是官方代码 感兴趣小伙伴可以对照看一下。

keras 已经实现

tf2.x 中 是tensorflow 2 版本

gelu 激活函数要求 tf>=2.4 否则使用入下代码代替gelu 激活函数

```python
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
```
  
csdn 博客链接 ：https://blog.csdn.net/qq_38676487/article/details/120705254?spm=1001.2014.3001.5501

其他问题私信：1308659229@qq.com
# ConvMixer
ConvMixer unofficial implementation

The pytorch version has been implemented.

nets is a refactored version and test is the official code. Interested friends can check it out.

keras has been implemented

tf2.x is the tensorflow 2 version

Gelu activation function requires tf>=2.4, otherwise use the following code instead of gelu activation function：

```python
def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf
```

