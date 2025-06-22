# Merged version of sonnet_generation.py combining LoRA fine-tuning and DPO training support.

import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
import sys
import sacrebleu

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from peft import get_peft_model, LoraConfig, TaskType

from datasets import SonnetsDataset, PairwiseSonnetsDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class SonnetGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        assert args.fine_tune_mode in ["full-model", "LoRA"]
        if args.fine_tune_mode == 'full-model':
            for param in self.gpt.parameters():
                param.requires_grad = True
        elif args.fine_tune_mode == 'LoRA':
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                target_modules=["query", "key", "value", "attention_dense"],
                fan_in_fan_out=True,
                task_type=TaskType.CAUSAL_LM
            )
            self.gpt = get_peft_model(self.gpt, peft_config)
            self.gpt.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        output = self.gpt(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden_states = output['last_hidden_state']
        logits = F.linear(hidden_states, self.gpt.get_input_embeddings().weight)
        return logits

    def get_device(self):
        for param in self.gpt.parameters():
            return param.device

    @torch.no_grad()
    def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
        token_ids = encoding.to(self.get_device())
        attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

        for _ in range(max_length):
            logits_sequence = self.forward(token_ids, attention_mask)
            logits_last_token = logits_sequence[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= top_p
            top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
            top_p_mask[..., 0] = True
            filtered_probs = sorted_probs * top_p_mask
            filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

            sampled_index = torch.multinomial(filtered_probs, 1)
            sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

            if sampled_token.item() == self.tokenizer.eos_token_id:
                break

            token_ids = torch.cat([token_ids, sampled_token], dim=1)
            attention_mask = torch.cat([
                attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())
            ], dim=1)

        generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
        return token_ids, generated_output

    def log_prob(self, prompt_ids, prompt_mask, target_ids, target_mask):
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, target_mask], dim=1)
        logits = self.forward(input_ids, attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = []
        for i in range(target_ids.size(0)):
            offset = prompt_ids.size(1)
            t_len = target_ids.size(1)
            lp = log_probs[i, offset-1:offset-1+t_len, :]
            lp = lp.gather(1, target_ids[i].unsqueeze(1)).squeeze(1)
            target_log_probs.append(lp.sum())
        return torch.stack(target_log_probs)

def dpo_loss(model, ref_model, batch, beta=0.1):
    device = next(model.parameters()).device
    for k in batch:
        batch[k] = batch[k].to(device)
    logp_w = model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['winner_ids'], batch['winner_mask'])
    logp_l = model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['loser_ids'], batch['loser_mask'])
    with torch.no_grad():
        logp_w_ref = ref_model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['winner_ids'], batch['winner_mask'])
        logp_l_ref = ref_model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['loser_ids'], batch['loser_mask'])
    loss = -torch.log(torch.sigmoid(beta * ((logp_w - logp_l) - (logp_w_ref - logp_l_ref))))
    return loss.mean()

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    dataset = SonnetsDataset(args.sonnet_path)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    held_out_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
            labels = b_ids[:, 1:].contiguous().flatten()
            loss = F.cross_entropy(logits, labels, reduction='mean')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch}: train loss :: {train_loss / len(dataloader):.3f}.")
        generate_and_evaluate(model, held_out_dataset, args)
        save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

def train_dpo(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    dataset = PairwiseSonnetsDataset(args.sonnet_path)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dataset.collate_fn)
    held_out_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args).to(device)
    ref_model = SonnetGPT(args).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(dataloader, desc=f'dpo-train-{epoch}', disable=TQDM_DISABLE):
            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"[DPO] Epoch {epoch}: train loss :: {train_loss / len(dataloader):.3f}.")
        generate_and_evaluate(model, held_out_dataset, args)
        save_model(model, optimizer, args, f'dpo_{epoch}_{args.filepath}')

def generate_and_evaluate(model, dataset, args):
    model.eval()
    generated, reference = [], []
    for batch in dataset:
        encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(model.get_device())
        _, output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
        generated.append(output.strip())
        reference.append(batch[1].strip())
        print(f"{batch[1]}{output.strip()}\n")
    evaluate_bleu(generated, reference)

def evaluate_bleu(generated, reference):
    bleu = sacrebleu.corpus_bleu(generated, [reference])
    print(f"[BLEU] corpus BLEU: {bleu.score:.2f}")
    return bleu.score

def save_model(model, optimizer, args, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
    parser.add_argument("--fine-tune-mode", type=str, choices=["full-model", "LoRA"], default="full-model")
    parser.add_argument("--dpo", action='store_true')
    return parser.parse_args()

def add_arguments(args):
    if args.model_size == 'gpt2':
        args.d = 768; args.l = 12; args.num_heads = 12
    elif args.model_size == 'gpt2-medium':
        args.d = 1024; args.l = 24; args.num_heads = 16
    elif args.model_size == 'gpt2-large':
        args.d = 1280; args.l = 36; args.num_heads = 20
    else:
        raise Exception(f"{args.model_size} is not supported.")
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.epochs}-{args.lr}-sonnet.pt"
    seed_everything(args.seed)
    if args.dpo:
        train_dpo(args)
    else:
        train(args)
