# Variable Batch Size Emulation

This python module can be used to train pytorch models with a *variable* batch size. That is, this modules enables training scripts to train with a dynamic batch size during training. The way the the batch size varies must be specified by a CSV file. Currently the module only works for PyTorch-XLA scripts, i.e. as used for TPU training.

### How to use

Assume a pytorch training script for TPU's (XLA) is given.

1. **Generate the batchsize trace:** The next steps requires a file *trace.csv*,
 specifying a batch size for each step, through all epochs. Specifically *trace.csv* should 
    be of format:
```
 step,batchsize
 0, 1024
 1, 1024
 2, 2048
 ...
```
The batchsize is the global batchsize and must be a multiple of the batch size specified in the data loader. It is the users responsibility that the trace is long enough

3. Use the custom **batch** sampler

```python
import variable_batch_size as vbs

train_sampler = vbs.CustomSampler(
  train_dataset,
  'trace.csv',
  num_replicas=global_world_size,
  rank=global_ordinal,
  minibatch_size=flags.batch_size)
```

4. Use the custom optimizer 

```python
internal_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
optimizer = vbs.LinearRuleOptimizer(internal_optimizer, ref_batchsize = 256)
```

The custom optimizer is aware of the step, and depending on it either does nothing, accumulate gradients, or synchs gradients across cores before updating the parameters

The custom optimizer takes care of zeroing the gradients so one can skip on calling `optimizer.zero_grad()`, it will not have any effect.

5. In the epoch loop, at each epoch call the following before iterating the dataloader.

```python
train_sampler.set_epoch(epoch)
optimizer.set_epoch(epoch)
```

5. If training using multiple TPU’s, set environment variables (depending on setup)

```bash
export MY_HOST_WORLD_SIZE=1
export MY_HOST_ORDINAL=0 
```

### Example MNIST

*variable_train_mnist.py* shows how it could work.

```bash
python3 variable_train_mnist.py
```
