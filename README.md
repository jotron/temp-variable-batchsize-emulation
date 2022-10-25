# Elastic Learning Rate Evaluation

This python module can be used to train pytorch models with a *variable* batch size. That is, this modules enables training scripts to train with a dynamic batch size during training. The way the the batch size varies can be specified by a CSV file. Currently the module only works for PyTorch-XLA scripts, i.e. as used for TPU training.

### How to use

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

### Running the test script

We provide a test script *variable_train_mnist.py* that uses our framework to train MNIST like [here](https://github.com/pytorch/xla/blob/master/test/test_profile_mp_mnist.py). To run the script on Cloud TPU instance with Pytorch XLA installed run the following:

```bash
git clone https://github.com/eth-easl/elastic-learning-rate-evaluation
cd elastic-learning-rate-evaluation/test
pip install tensorboardX pandas
python variable_train_mnist.py
```

