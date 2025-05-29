import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from data import ZuCo_dataset
import copy
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load

wer_metric = load("wer")

class DeepSeekLoRAModel(nn.Module):
    def __init__(self, deepseek_model_name, t5_model_name, output_dim, eeg_dim, device):
        super().__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)
        self.t5_dim = self.t5_encoder.config.d_model
        self.deepseek = AutoModelForCausalLM.from_pretrained(deepseek_model_name)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        self.deepseek = get_peft_model(self.deepseek, lora_config)
        self.deepseek_dim = self.deepseek.config.hidden_size
        self.eeg_processor = nn.Linear(eeg_dim, 128) if eeg_dim else None
        input_dim = self.t5_dim + self.deepseek_dim + (128 if eeg_dim else 0)
        self.fc = nn.Linear(input_dim, output_dim)
        self.to(device)

    def forward(self, input_ids, attention_mask, eeg=None, deepseek_input_ids=None, deepseek_attention_mask=None):
        t5_outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        t5_emb = t5_outputs.last_hidden_state[:, 0, :]
        deepseek_outputs = self.deepseek(deepseek_input_ids, attention_mask=deepseek_attention_mask, output_hidden_states=True)
        deepseek_emb = deepseek_outputs.hidden_states[-1][:, 0, :]
        features = [t5_emb, deepseek_emb]
        if self.eeg_processor and eeg is not None:
            eeg_features = self.eeg_processor(eeg)
            features.append(eeg_features)
        combined_features = torch.cat(features, dim=-1)
        output = self.fc(combined_features)
        return output

def word_to_embedding(word, tokenizer, t5_encoder, device):
    encoding = tokenizer(word, return_tensors='pt', max_length=10, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = t5_encoder(input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze()

def train_model(model, dataloaders, device, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best='./checkpoints/best_model.pt', checkpoint_path_last='./checkpoints/last_model.pt'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 50)
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for batch in tqdm(dataloaders[phase]):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                eeg = batch.get('eeg', None)
                if eeg is not None:
                    eeg = eeg.to(device)
                target_words = batch['target_word']
                target_embeddings = torch.stack([word_to_embedding(word, t5_tokenizer, model.t5_encoder, device) for word in target_words]).to(device)
                deepseek_encoding = deepseek_tokenizer(' '.join(target_words), return_tensors='pt', max_length=128, padding='max_length', truncation=True)
                deepseek_input_ids = deepseek_encoding['input_ids'].to(device)
                deepseek_attention_mask = deepseek_encoding['attention_mask'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input_ids, attention_mask, eeg, deepseek_input_ids, deepseek_attention_mask)
                    loss = criterion(outputs, target_embeddings)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_ids.size(0)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.0f}')
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'Updated best checkpoint: {checkpoint_path_best}')
        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.0f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'Updated last checkpoint: {checkpoint_path_last}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    task_name = 'task1'
    deepseek_model_name = 'DeepSeek/DeepSeek-R-1-8B'  # Замените на точное название
    t5_model_name = 't5-large'
    batch_size = 1
    num_epochs_step1 = 10
    num_epochs_step2 = 5
    step1_lr = 1e-3
    step2_lr = 1e-4
    subject_choice = 'ALL'
    eeg_type_choice = 'GD'
    bands_choice = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']
    train_input = 'EEG'
    dataset_setting = 'unique_sent'
    save_path = './checkpoints'
    save_name = f'{task_name}_DeepSeekLoRA_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    save_path_best = os.path.join(save_path, 'best')
    save_path_last = os.path.join(save_path, 'last')
    os.makedirs(save_path_best, exist_ok=True)
    os.makedirs(save_path_last, exist_ok=True)
    checkpoint_path_best = os.path.join(save_path_best, f'{save_name}.pt')
    checkpoint_path_last = os.path.join(save_path_last, f'{save_name}.pt')
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device {device}')
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path = '/data/johj/ZuCo_data/task1-SR/task1_source.pkl'
        with open(dataset_path, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_name)
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', t5_tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', t5_tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print(f'[INFO] train_set size: {len(train_set)}')
    print(f'[INFO] dev_set size: {len(dev_set)}')
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        'dev': DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=4)
    }
    eeg_dim = 105 * len(bands_choice)
    model = DeepSeekLoRAModel(deepseek_model_name, t5_model_name, output_dim=1024, eeg_dim=eeg_dim, device=device)
    model = torch.nn.DataParallel(model)
    for name, param in model.named_parameters():
        if 't5_encoder' in name or ('deepseek' in name and 'lora' not in name):
            param.requires_grad = False
    print('Step 1: Training LoRA and FC layers')
    optimizer_step1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr)
    scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)
    criterion = nn.MSELoss()
    model = train_model(model, dataloaders, device, criterion, optimizer_step1, scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best=checkpoint_path_best, checkpoint_path_last=checkpoint_path_last)
    for name, param in model.named_parameters():
        if 't5_encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print('Step 2: Fine-tuning LoRA, DeepSeek, and FC layers')
    optimizer_step2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=step2_lr)
    scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)
    model = train_model(model, dataloaders, device, criterion, optimizer_step2, scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best=checkpoint_path_best, checkpoint_path_last=checkpoint_path_last)
    def embedding_to_word(embedding, tokenizer, t5_encoder, device, top_k=1):
        encoding = tokenizer('word:', return_tensors='pt', max_length=10, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            vocab_embeddings = t5_encoder(input_ids=input_ids.repeat(tokenizer.vocab_size, 1), attention_mask=attention_mask.repeat(tokenizer.vocab_size, 1)).last_hidden_state[:, 0, :]
            cos_sim = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), vocab_embeddings, dim=-1)
            top_k_indices = cos_sim.topk(top_k).indices
            return tokenizer.decode(top_k_indices[0])
    model.eval()
    predicted_words = []
    target_words = []
    with torch.no_grad():
        for batch in dataloaders['dev']:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            eeg = batch.get('eeg', None)
            if eeg is not None:
                eeg = eeg.to(device)
            target_words_batch = batch['target_word']
            deepseek_encoding = deepseek_tokenizer(' '.join(target_words_batch), return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            deepseek_input_ids = deepseek_encoding['input_ids'].to(device)
            deepseek_attention_mask = deepseek_encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask, eeg, deepseek_input_ids, deepseek_attention_mask)
            predicted_word = embedding_to_word(outputs[0], t5_tokenizer, model.t5_encoder, device)
            predicted_words.append(predicted_word)
            target_words.append(target_words_batch[0])
    bleu_scores = [sentence_bleu([tw.split()], pw.split(), weights=(1.0,)) for tw, pw in zip(target_words, predicted_words)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU-1 score: {avg_bleu}")
    wer_score = wer_metric.compute(predictions=predicted_words, references=target_words)
    print(f"WER score: {wer_score}")
