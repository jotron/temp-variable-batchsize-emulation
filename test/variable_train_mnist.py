"""
Reference Implementation on MNIST.

Based on: github.com/pytorch/xla/blob/master/test/test_train_mp_mnist.py
Copied on 23/10/22. 

Special dependencies: 
  - tensorboardX (pip install tensorboardX)
  - pandas

Run with:
  python3 variable_train_mnist.py
"""

# Add parent dir to path to import module
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

import variable_batch_size as vbs

FLAGS = dict(
    datadir='/tmp/mnist-data',
    batch_size=32,
    momentum=0.5,
    lr=0.01,
    num_cores=8,
    num_workers=4,
    num_epochs=5,
    log_steps=5,
    log_all = False, # Log every step AND every accumulation step.
    target_accuracy=97.0,
    trace='test_trace.csv',
    ref_batchsize=128,
    tidy=False,
    metrics_debug=False,
    fake_data=False,
    drop_last=False,
    logdir='.',
    async_closures=None)

import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils


class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train_mnist(flags, **kwargs):
  torch.manual_seed(1)

  if flags['fake_data']:
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(flags['batch_size'], 1, 28,
                          28), torch.zeros(flags['batch_size'],
                                           dtype=torch.int64)),
        sample_count=60000 // flags['batch_size'] // xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(flags['batch_size'], 1, 28,
                          28), torch.zeros(flags['batch_size'],
                                           dtype=torch.int64)),
        sample_count=10000 // flags['batch_size'] // xm.xrt_world_size())
  else:
    train_dataset = datasets.MNIST(
        os.path.join(flags['datadir'], str(xm.get_ordinal())),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST(
        os.path.join(flags['datadir'], str(xm.get_ordinal())),
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]))

    train_sampler = vbs.CustomSampler(
      train_dataset,
      flags['trace'],
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      minibatch_size=flags['batch_size'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=flags['num_workers'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags['batch_size'],
        drop_last=flags['drop_last'],
        shuffle=False,
        num_workers=flags['num_workers'])

  device = xm.xla_device()
  model = MNIST().to(device)
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags['logdir'])
  internal_optimizer = optim.SGD(model.parameters(), lr=flags['lr'], momentum=flags['momentum'])
  optimizer = vbs.LinearRuleOptimizer(internal_optimizer, train_sampler, ref_batchsize = 128, log_steps=flags['log_steps'] if not flags['log_all'] else 1)
  loss_fn = nn.NLLLoss()

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    model.train()
    for step, (data, target) in enumerate(loader):
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      tracker.add(flags['batch_size'])
      if flags['log_all']:
        xm.add_step_closure(
            _train_update,
            args=(device, step, loss, tracker, epoch, writer),
            run_async=flags['async_closures'])

  def test_loop_fn(loader):
    total_samples = 0
    correct = 0
    model.eval()
    for data, target in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  xm.rendezvous("Training Start")
  for epoch in range(1, flags['num_epochs'] + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_sampler.set_epoch(epoch-1)
    optimizer.set_epoch(epoch-1)
    train_loop_fn(train_device_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    accuracy = test_loop_fn(test_device_loader)
    xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
        epoch, test_utils.now(), accuracy))
    max_accuracy = max(accuracy, max_accuracy)
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={'Accuracy/test': accuracy},
        write_xla_metrics=True)
    if flags['metrics_debug']:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  torch.set_default_tensor_type('torch.FloatTensor')
  accuracy = train_mnist(flags)
  if flags['tidy'] and os.path.isdir(flags['datadir']):
    shutil.rmtree(flags['datadir'])
  if accuracy < flags['target_accuracy']:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  flags['target_accuracy']))
    sys.exit(21)


if __name__ == '__main__':
  os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS['num_cores'])
