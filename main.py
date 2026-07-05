import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from config import Config
from data import get_dataloaders
from model import Transformer
from train import train_epoch, load_checkpoint, save_checkpoint, LabelSmoothingLoss, NoamOpt
from evaluate import evaluate_bleu, generate_translation, plot_attention_heatmap
from utils import logger
import os

def main():
    parser = argparse.ArgumentParser(description="Transformer Implementation (Attention is All You Need)")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "overfit"], default="train", help="Mode to run the script in")
    args = parser.parse_args()
    
    config = Config()
    logger.info(f"Running in mode: {args.mode}")
    logger.info(f"Using device: {config.train.device}")
    
    train_dl, val_dl, src_tok, tgt_tok = get_dataloaders(config)
    
    config.model.vocab_size = src_tok.get_vocab_size() # Ensure they match
    model = Transformer(config.model).to(config.train.device)
    
    pad_id = src_tok.token_to_id('[PAD]')
    criterion = LabelSmoothingLoss(config.model.vocab_size, padding_idx=pad_id, smoothing=config.model.label_smoothing)
    
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOpt(config.model.d_model, config.train.learning_rate_scale, config.train.warmup_steps, optimizer)
    scaler = GradScaler('cuda', enabled=(config.train.device == 'cuda'))
    
    start_epoch = load_checkpoint(model, scheduler, config)
    
    if args.mode == "train":
        for epoch in range(start_epoch, config.train.num_epochs):
            logger.info(f"Starting Epoch {epoch}")
            loss = train_epoch(model, train_dl, scheduler, criterion, scaler, config, epoch, pad_id)
            logger.info(f"Epoch {epoch} finished. Average Loss: {loss:.4f}")
            
            save_checkpoint(model, scheduler, epoch, config)
            evaluate_bleu(model, val_dl, src_tok, tgt_tok, config)
            
    elif args.mode == "eval":
        evaluate_bleu(model, val_dl, src_tok, tgt_tok, config)
        
        # Sample qualitative test
        sample_sentence = "A dog is running in the snow."
        translated, attn_weights = generate_translation(model, sample_sentence, src_tok, tgt_tok, config)
        logger.info(f"Source: {sample_sentence}")
        logger.info(f"Translated: {translated}")
        
        # Heatmap
        src_tokens = src_tok.encode(sample_sentence).tokens
        tgt_tokens = tgt_tok.encode(translated).tokens
        plot_attention_heatmap(attn_weights, src_tokens, tgt_tokens)
        
    elif args.mode == "overfit":
        logger.info("Running overfit sanity check on a single batch...")
        config.train.log_every_steps = 1
        # get one batch
        src, tgt = next(iter(train_dl))
        
        for step in range(200):
            model.train()
            src = src.to(config.train.device)
            tgt = tgt.to(config.train.device)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            out, _ = model(src, tgt_input)
            loss = criterion(out.contiguous().view(-1, out.size(-1)), 
                             tgt_expected.contiguous().view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(scheduler.optimizer)
            scaler.update()
            scheduler.optimizer.zero_grad()
            
            if step % 20 == 0:
                logger.info(f"Step {step} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
