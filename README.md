# DeepDream
> deep dream 透過增強某些在 DNN layer 的 weight，讓人類知道究竟機器學習到了什麼

## Instruction
According to [Inceptionism: Going Deeper into Neural Networks](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)

1. Lower layers tend to produce strokes or simple ornament-like patterns, because those layers are sensitive to basic features such as edges and their orientations
2. Higher-level layers, which identify more sophisticated features in images, complex features or even whole objects tend to emerge

## How to use
**Only in linux**

For DeepDream:

```bash
python3 deepdream.py -s hello.jpg
```

For DeepStyle:

```bash
python3 deepdream.py -s hello.jpg -d fruit.jpeg
```

## Reference
[google github](https://github.com/google/deepdream)
[other info](https://hackmd.io/r90FWvR0RQyHnkLS5ooo5g)
