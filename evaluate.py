import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils import logger
import os

def generate_translation(model, src_sentence, src_tokenizer, tgt_tokenizer, config, max_len=50):
    model.eval()
    
    # Tokenize input
    src_encoded = src_tokenizer.encode(src_sentence).ids
    sos_id = src_tokenizer.token_to_id('[SOS]')
    eos_id = src_tokenizer.token_to_id('[EOS]')
    src_ids = [sos_id] + src_encoded + [eos_id]
    
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(config.train.device)
    
    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor)
        enc_out = model.encoder(src_tensor, src_mask)
        
        tgt_ids = [sos_id]
        
        for i in range(max_len):
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(config.train.device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            
            dec_out, attn_weights = model.decoder(tgt_tensor, enc_out, src_mask, tgt_mask)
            out = model.fc_out(dec_out)
            
            # Get next token
            next_token = out.argmax(2)[:, -1].item()
            tgt_ids.append(next_token)
            
            if next_token == eos_id:
                break
                
    generated_text = tgt_tokenizer.decode(tgt_ids)
    return generated_text, attn_weights

def plot_attention_heatmap(attn_weights, src_tokens, tgt_tokens, head_idx=0, layer_idx=-1, save_path="attention_heatmap.png"):
    # attn_weights is a list of tensors for each layer: [batch_size, n_heads, tgt_len, src_len]
    attn = attn_weights[layer_idx][0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap='viridis')
    plt.xlabel('Source (English)')
    plt.ylabel('Target (German)')
    plt.title(f'Attention Heatmap (Layer {layer_idx}, Head {head_idx})')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved attention heatmap to {save_path}")

def evaluate_bleu(model, dataloader, src_tokenizer, tgt_tokenizer, config):
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu not installed. Skip BLEU evaluation.")
        return 0.0

    model.eval()
    references = []
    hypotheses = []
    
    # Just take a subset for quick evaluation if dataset is large
    eval_limit = 200
    
    logger.info("Evaluating BLEU score...")
    for idx, item in enumerate(dataloader.dataset.data):
        if idx >= eval_limit: break
        
        src_text = item['translation']['en']
        ref_text = item['translation']['de']
        
        hyp_text, _ = generate_translation(model, src_text, src_tokenizer, tgt_tokenizer, config)
        
        references.append([ref_text])
        hypotheses.append(hyp_text)
        
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    logger.info(f"BLEU Score: {bleu.score:.2f}")
    return bleu.score
