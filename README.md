# Elastic Learning Rate Evaluation

This python module can be used to train pytorch models with a *variable* batch size. That is, this modules enables training scripts to train with a dynamic batch size during training. The way the the batch size varies can be specified by a CSV file. Currently the module only works for PyTorch-XLA scripts, i.e. as used for TPU training.

### How to use.

Assume a pytorch training script for TPU's (XLA) is given.

1. **Generate the batchsize trace:** Create a file *trace.csv*, specifying a batch size for each step, through all epochs. Specifically *trace.csv* should be of format:
```
 step,batchsize
 0, 1024
 1, 1024
 2, 2048
 ...
```
The batchsize is the global batchsize and must be a multiple of the batch size specified in the data loader. It is the users responsibility that the trace is long enough to train the model to completion.

3. Use the custom **batch** sampler

```python
import variable_batch_size as vbs
train_sampler = vbs.CustomSampler(
  dataset_size=len(train_dataset),
  trace='trace.csv',
  num_replicas=WORLD_SIZE,
  rank=ORDINAL,
  minibatch_size=flags.batch_size)
train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_sampler=train_sampler)
```

4. Use the custom optimizer 

```python
internal_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=flags.momentum)
optimizer = vbs.LinearRuleOptimizer(
  internal_optimizer, 
  train_sampler, 
  ref_batchsize = 128, 
  log_steps=flags.log_steps)
```

The custom optimizer is aware of the step, and depending on it either does nothing, accumulate gradients, or synchronizes gradients across cores before updating the parameters. The custom optimizer takes care of zeroing the gradients so one can skip on calling `optimizer.zero_grad()` as it will not have any effect.

5. In the epoch loop, at each epoch call the following before iterating the dataloader.

```python
train_sampler.set_epoch(epoch)
optimizer.set_epoch(epoch)
```

### How to run the test script.

We provide a test script *variable_train_mnist.py* that uses our framework to train MNIST like [here](https://github.com/pytorch/xla/blob/master/test/test_profile_mp_mnist.py). To run the script on Cloud TPU instance with Pytorch XLA installed run the following:

```bash
git clone https://github.com/eth-easl/elastic-learning-rate-evaluation
cd elastic-learning-rate-evaluation/test
pip install tensorboardX pandas
python variable_train_mnist.py
```

### How it works.

The idea is that your script iterates through essentially the same data as before and all the logic is hidden in the optimizer. Depending on the batch size at a certain step the optimizer either updates the weights or accumulates the gradients to get a larger batch size. If for example the local batch size of your dataloader is 32, we train with a single core, and the trace starts with [64, 96, …] the first few invocations of optimizer.step() will do the following:

*1: Accumulate 2: Update 3: Accumulate 4: Accumulate 5: Update …*

Multi-core training works similarly except that depending on the batch size not all cores may accumulate for the same number of “substeps”.

### Learning rate scaling

As the batch size changes you may want to change the learning rate, e.g. linearly. To facilitate this a number of learning rate scaling rules are already provided in the form of optimizer subclasses:

- Linear Scaling Rule (Default): *LinearRuleOptimizer*
- Root Scaling Rule: *RootRuleOptimizer*
- AdaScale Rule: *AdaScaleOptimizer*
  - Note that this optimizer requires further changes to your training script (See the [original paper](https://arxiv.org/abs/2007.05105)).
