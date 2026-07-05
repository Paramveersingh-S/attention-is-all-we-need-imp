import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import time
from .config import Config
from .utils import logger
from .model import Transformer

class NoamOpt:
    """Optim wrapper that implements rate."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.classes
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.classes - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.requires_grad_(False))

def train_epoch(model, dataloader, optimizer, criterion, scaler, config, epoch, pad_id):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(config.train.device)
        tgt = tgt.to(config.train.device)
        
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        
        with autocast(enabled=(config.train.device == 'cuda')):
            out, _ = model(src, tgt_input)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), 
                             tgt_expected.contiguous().view(-1))
            loss = loss / config.train.gradient_accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.train.gradient_accumulation_steps == 0:
            scaler.step(optimizer.optimizer)
            scaler.update()
            optimizer.optimizer.zero_grad()
            
        total_loss += loss.item() * config.train.gradient_accumulation_steps
        
        if batch_idx % config.train.log_every_steps == 0:
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch} | Step {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss.item() * config.train.gradient_accumulation_steps:.4f} | "
                        f"LR: {optimizer._rate:.6f} | "
                        f"Tokens/s: {src.size(0) * src.size(1) / elapsed:.1f}")
            start_time = time.time()
            
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, config, path="checkpoint.pt"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict(),
        'step': optimizer._step
    }
    torch.save(checkpoint, os.path.join(config.train.checkpoint_dir, path))
    logger.info(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, config, path="checkpoint.pt"):
    full_path = os.path.join(config.train.checkpoint_dir, path)
    if os.path.exists(full_path):
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer._step = checkpoint['step']
        logger.info(f"Resumed from checkpoint: {full_path} (Epoch {checkpoint['epoch']})")
        return checkpoint['epoch'] + 1
    return 0
