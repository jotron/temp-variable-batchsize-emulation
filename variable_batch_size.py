"""
Variable Batch Size Module
"""

import torch.distributed as dist
import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
try:
  import torch_xla.core.xla_model as xm
except:
  print("    Running without torch_xla")

class CustomSampler(DistributedSampler):
  """
  Sampler to be used in combination with torch.utils.data.DataLoader.
  The sampler returns a batches of indices, and the dataloader is responsible for
  fetching indices from the dataset.
  Note: This implementation drops the last sampels of the epoch that don't suffice to form an entire batch.
  """
  def __init__(self, dataset, trace, seed=42, num_replicas=None, rank=None, 
    minibatch_size=32, verbose=True):
    """
    Loads the trace from a CSV file.

    dataset: Dataset used for sampling
    trace: path or list or INT
      if path: loads trace CSV file.
      if list: list[i] specifies batchsize at step i.
      if INT: batchsize is constant.
    seed (optional): Random seed to use for shuffling
    num_replicas (optional): Number of processes participating in
        distributed training.
    rank (optional): Rank of the current process within num_replicas.
    minibatch_size (optional): local batch size to use for accumulation.
    """

    super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
    self.seed = seed
    self.minibatch_size = minibatch_size
    self.verbose = verbose
    
    # Load trace
    if isinstance(trace, str):
      df = pd.read_csv(trace)
      self.trace = df.batchsize.to_list()
    elif isinstance(trace, list):
      self.trace = trace
    elif isinstance(trace, int):
      self.trace = [trace] * 2000000
    if verbose and rank==0:
        print(f"    MySampler (rank {rank}): Loaded trace starting with {self.trace[0]}, {self.trace[1]},...")

    # Check trace is valid
    tot_samples = 0
    for s in self.trace:
        if s%minibatch_size != 0:
            raise ValueError(f"Every Sample batch size should be multiple of {minibatch_size}")
        tot_samples += s

    # Check length of trace
    self.minibatches_per_epoch = len(self.dataset) // minibatch_size
    self.max_epoch = tot_samples // (self.minibatches_per_epoch * minibatch_size)
    if verbose and rank==0:
        print(f"    MySampler (rank {rank}): trace_samples={tot_samples}, minibatches_per_epoch={self.minibatches_per_epoch} => max_epoch=~{self.max_epoch}")

  def set_epoch(self, epoch: int) -> None:
    if (epoch > self.max_epoch):
      print(f"    Custom Sampler Warning: epoch > max_epoch")
    self.epoch = epoch

  def get_batch_assignments(self, epochtrace):
    """
    @parameter epochtrace: Trace of batchsizes for this epoch. 
      Sum should be smaller equal than minibatches_per_epoch*minibatch_size
    Returns a list of lists where sublist j contains the minibatche-indices contributing to step j from this rank.
    A sublist with more than one element implies accumulation.
    """
    batches = []
    step_start = 0
    all_minibatches = list(range(self.minibatches_per_epoch))
    steps_per_epoch = len(epochtrace)
    for i in range(steps_per_epoch):
      minibatches_per_step = epochtrace[i] // self.minibatch_size
      step_end = step_start + minibatches_per_step
      local_minibatches = all_minibatches[(step_start + self.rank) : step_end : self.num_replicas]
      batches.append(local_minibatches)
      step_start = step_end

    return batches

  def get_epoch_range(self, epoch):
    """
    Returns the start and end index of this epoch in the trace.
    """
    start, end, epochsum, e = 0, 0, 0, -1
    # INVARIANT: [start..end-1] is the partition for epoch *epoch*
    while (e < epoch):
      start = end
      while (epochsum + self.trace[end] <= self.minibatch_size*self.minibatches_per_epoch):
        epochsum += self.trace[end]
        end += 1
      e += 1
      epochsum = 0
    return start, end

  def get_epochtrace(self, epoch):
    """
    Returns the list of batchsizes for this epoch.
    """
    start, end = self.get_epoch_range(epoch)
    trace = self.trace[start:end]
    assert(sum(trace) <= self.minibatch_size*self.minibatches_per_epoch)
    return trace

  def __bool__(self):
    return True

  def __iter__(self):
    """
    Returns a subset of [0, len(dataset)-1] partitioned into minibatches. Every index occurs at most once, except index 0, 
    which is used when the core doesn't actually participate in the step.
    The set of iterators for all ranks forms a partition of [0, len(dataset)-1]. 
    The sum of globally seen minibatches over a step including accumulation is equal 
    to the desired global batch size at step i.

    Make sure self.epoch is set correctly. 
    """
    g = torch.Generator()
    g.manual_seed(self.seed + self.epoch)

    # first permute, and then select
    indices = torch.randperm(len(self.dataset), generator=g).tolist()
    indices = indices[:self.minibatches_per_epoch*self.minibatch_size]

    # Batchsizes for each step of the epoch
    epochtrace = self.get_epochtrace(self.epoch)
    # Batches assigned to this rank for each step
    batchlist = self.get_batch_assignments(epochtrace)
    # Batches to concrete indices
    steplist = []
    for batches in batchlist:
      if (batches != []):
        for bi in batches:
          steplist.append(indices[bi*self.minibatch_size:(bi+1)*self.minibatch_size])
      else:
        steplist.append([0 for _ in range(self.minibatch_size)])

    #if self.verbose:
      #print(f"Batch Assignment for first step worker {self.rank}: {batchlist[0]}")
    return iter(steplist)


class CustomOptimizer(object):
  def __init__(self, optimizer, sampler, log_steps=None, reduction='mean'):
    """
    optimizer: internal optimizer to use, e.g. torch.optim.SGD
    sampler: the CustomSampler instance used.
    log_steps: integer, logging interval.
    reduction: 'mean' or 'sum', how to combine gradients.

    The way this class is currently implemented, the optimizer is strongly coupled with the sampler. 
    This was originally done to avoid code duplication.
    
    """
    self.optimizer = optimizer
    self.sampler = sampler
    self.set_epoch(0)
    self.log_steps = log_steps
    self.reduction = reduction
    assert(reduction in ['sum', 'mean'])
    if (reduction == 'sum'):
      xm.master_print(f"Using {reduction} reduction!")

  def set_epoch(self, epoch):
    # Trace
    self.epochtrace = self.sampler.get_epochtrace(epoch)
    # Scaling Factor for Each Step
    self.divisors = [t // self.sampler.minibatch_size for t in self.epochtrace]
    # Batch Assignments
    self.schedule = self.sampler.get_batch_assignments(self.epochtrace)
    self.step_index = 0
    self.substep_index = 0 # for accumulation
    self.optimizer.zero_grad()

  def step(self):
    """
    Returns true if self.optimizer.step was called, i.e. the weights were updated and 
    not only gradients accumulated.
    """
    subbatches = self.schedule[self.step_index]
    # Ignore dummy batches
    if subbatches == []:
      # This is the case when the rank is not involved in the step 
      # (because global batchsize very small)
      self.optimizer.zero_grad()
    # Tyme to sync
    if subbatches == [] or self.substep_index == len(subbatches)-1:
      # Scale and sync Gradients
      grads = self._fetch_gradients()
      scale = 1.0 if self.reduction=='sum' else 1.0/self.divisors[self.step_index]
      all_reduce_tensors(grads, scale)

      # Adjust LR
      ref_lr = self.get_lr()
      self.adapt_lr()
      self.optimizer.step()
      if self.log_steps is not None and (self.step_index % self.log_steps == 0): 
        xm.master_print(f"    CustomOptimizer: Step={self.step_index}, bs={self.epochtrace[self.step_index]}, lr={self.get_lr()}")

      self.optimizer.zero_grad()
      self.set_lr(ref_lr) # Set lr back to reference lr.
      self.step_index += 1
      self.substep_index = 0
      xm.mark_step()
      return True
    # Do nothing, and just let gradients accumulate.
    else:
      self.substep_index += 1
      return False

  def get_lr(self):
    values = tuple(param_group['lr'] for param_group in self.optimizer.param_groups)
    return values

  def set_lr(self, values):
    for i, data in enumerate(zip(self.optimizer.param_groups, values)):
      param_group, lr = data
      param_group['lr'] = lr

  def get_bs(self):
    """
    Returns batchsize at current step_index, 0 if past epoch.
    """
    if self.step_index < len(self.epochtrace):
      return self.epochtrace[self.step_index]
    else: return 0

  def adapt_lr(self):
    pass

  def state_dict(self):
    pass

  def load_state_dict(self):
    pass

  def zero_grad(self):
    # No need to call this, as it is done automatically and dynamically.
    pass

  def _fetch_gradients(self):
    """
    Provides list of gradient tensors.
    """
    gradients = []
    for param_group in self.optimizer.__getstate__()['param_groups']:
      for group, params in param_group.items():
        if group == 'params':
          for p in params:
            if isinstance(p, torch.Tensor) and p.grad is not None:
              gradients.append(p.grad.data)
    return gradients


class LinearRuleOptimizer(CustomOptimizer):
  def __init__(self, optimizer, sampler, ref_batchsize, log_steps=None):
    super(LinearRuleOptimizer, self).__init__(optimizer, sampler, log_steps)
    self.ref_batchsize = ref_batchsize

  def F(ref_lr, batchsize, ref_batchsize):
    return batchsize/ref_batchsize * ref_lr
  
  def adapt_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()
    lr = tuple(LinearRuleOptimizer.F(rlr, bs, self.ref_batchsize) for rlr in ref_lr)
    self.set_lr(lr)

  def current_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()[0]
    return LinearRuleOptimizer.F(ref_lr, bs, self.ref_batchsize)


class ConstantOptimizer(CustomOptimizer):
  def __init__(self, optimizer, sampler, ref_batchsize, log_steps=None):
    super(ConstantOptimizer, self).__init__(optimizer, sampler, log_steps)
    self.ref_batchsize = ref_batchsize

  def adapt_lr(self):
    return

  def current_lr(self):
    ref_lr = self.get_lr()[0]
    return ref_lr


class LinearRuleMomentumOptimizer(CustomOptimizer):
  """
  The gradients across subbatches are no longer averaged but summed over subbatches instead.
  The learning rate is the linear scaled learning rate for a subbatch of 32.
  As the scale of the gradients changes, the scale of the weight decay must also!
  Only works for single Parameter Group, very coupled."""
  def __init__(self, optimizer, sampler, ref_batchsize, weight_decay=1e-4, log_steps=None):
    super(LinearRuleMomentumOptimizer, self).__init__(optimizer, sampler, log_steps, reduction='sum')
    self.ref_batchsize = ref_batchsize
    self.weight_decay=weight_decay

  def F(ref_lr, batchsize, ref_batchsize):
    return batchsize/ref_batchsize * ref_lr
  
  def adapt_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()
    lr = tuple(LinearRuleMomentumOptimizer.F(rlr, self.sampler.minibatch_size, self.ref_batchsize) for rlr in ref_lr)
    self.set_lr(lr)
    # TRICKY: Need to correct weight decay!!
    for group in self.optimizer.param_groups:
      group['weight_decay'] = self.weight_decay*self.divisors[self.step_index]

  def current_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()[0]
    return LinearRuleMomentumOptimizer.F(ref_lr, bs, self.ref_batchsize)


class RootRuleOptimizer(CustomOptimizer):
  def __init__(self, optimizer, sampler, ref_batchsize, log_steps=None):
    super(RootRuleOptimizer, self).__init__(optimizer, sampler, log_steps)
    self.ref_batchsize = ref_batchsize

  def F(ref_lr, batchsize, ref_batchsize):
    return (batchsize/ref_batchsize)**0.5 * ref_lr
  
  def adapt_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()
    lr = tuple(RootRuleOptimizer.F(rlr, bs, self.ref_batchsize) for rlr in ref_lr)
    self.set_lr(lr)

  def current_lr(self):
    bs = self.get_bs()
    ref_lr = self.get_lr()[0]
    return RootRuleOptimizer.F(ref_lr, bs, self.ref_batchsize)


def init_group(local_ordinal, cores_per_host=8):
  """
  Initialize group for inter-device synchronization.
  @return global ordinal, global world size
  """ 
  host_ordinal = int(os.environ.get('MY_HOST_ORDINAL', default="0"))
  host_world_size = int(os.environ.get('MY_HOST_WORLD_SIZE', default="1"))
  global_ordinal = cores_per_host*host_ordinal + local_ordinal
  global_world_size = host_world_size * cores_per_host

  if (host_world_size > 1 and local_ordinal == 0):
    init_method = 'tcp://' + os.environ.get('MY_DIST_ROOT')
    dist.init_process_group('gloo', init_method=init_method, rank=host_ordinal, world_size=host_world_size)
    print(f"    ---- Inter Node Process Group Initialized ----")

  return (global_ordinal, global_world_size)


def all_reduce_tensors(tensors, scale):
  """
  Perform global two-level all reduce.
  Args:
    tensors: List of `torch.Tensor`
    scale (float): scaling factor
  """
  # Locally Reduce
  xm.all_reduce(reduce_type=xm.REDUCE_SUM, inputs=tensors, scale=scale)
  
  # Inter-Node Reduce rank 0
  host_world_size = int(os.environ.get('MY_HOST_WORLD_SIZE', default="1"))
  if host_world_size > 1:
    if xm.get_ordinal() == 0:
      xm.mark_step()
      cpu_tensors = [tensor.cpu() for tensor in tensors]
      dist.all_reduce_coalesced(cpu_tensors, op=dist.ReduceOp.SUM)
      for i in range(len(tensors)):
        tensors[i].copy_(cpu_tensors[i])
      
    # Broadcast to other cores
    # Other cores don't contribute
    if xm.get_ordinal() != 0:
      for tensor in tensors:
        tensor.fill_(0.0)
    xm.all_reduce(xm.REDUCE_SUM, tensors)


def all_reduce_tensors_mesh(tag, data, scale=1.0):
  """
  Perform local all-reduce on CPU memory.
  Args:
    data: List of `torch.Tensor`
    scale (float): scaling factor
  """
  reduce_fn = lambda x: np.sum(x, axis=0)
  x = xm.mesh_reduce('m', data, reduce_fn)
  return scale * x
