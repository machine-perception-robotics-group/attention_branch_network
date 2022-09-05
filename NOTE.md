# NOTE


## `target.cuda(async=True)`

### Duplication

Since torch 0.4, `async` is deprecated in cuda because it is a reserved word in Python >= 3.7, `non_blocking` should be used instead.

Please see [this GitHub issue](https://github.com/quark0/darts/issues/64) for more details.

### Function of non_blocking (or async)

If the next operation depends on your data, you won’t notice any speed advantage.
However, if the asynchronous data transfer is possible, you might hide the transfer time in another operation.

Plase see [this PyTorch forum](https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234) for more details.


## `.view()` operation

If we use view operation, the elements should be placed in the order of the array in the memory.
Otherwise, we need to apply `.contiguous()` before `.view()` operation.


## Problem on Tensor.data[0]

"invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number"

**torch 0.4.0**

```python
losses.update(loss.data[0], inputs.size(0))
top1.update(prec1[0], inputs.size(0))
top5.update(prec5[0], inputs.size(0))
```

**torch 1.13.0**

```python
losses.update(loss.item(), inputs.size(0))
top1.update(prec1.item(), inputs.size(0))
top5.update(prec5.item(), inputs.size(0))
```


