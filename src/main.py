import re
from tqdm import tqdm
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import GPT2Config
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import ast
import math
from transformers import get_cosine_schedule_with_warmup
import pandas as pd

from titles.TitleDataset import TitleDataset
from recipes.RecipeDataset import RecipeDataset

def preprocess_ingredient(ingredient):
    ingredient = re.sub(r"""[^a-zA-Z0-9\s/\.\(\)]""","",ingredient,flags=re.VERBOSE)

    ingredient = re.sub(r"\s+", " ", ingredient).strip().lower()
    replacements = {
        r"\bc\.": "cup",
        r"\bpkg\.": "package",
        r"\bml\.": "milliliter",
        r"\bgr?": "gram",
        r"\btbsp\.": "tablespoon",
        r"\btsp\.": "teaspoon",
        r"\bkg\.": "kilogram",
        r"\boz\.": "ounce",
        r"\blb\.": "pound"
    }
    for pattern, replacement in replacements.items():
        ingredient = re.sub(pattern, replacement, ingredient)

    return ingredient



def train_model(model, num_epochs, optimizer, train_loader, valid_loader, tokenizer, device, checkpoint_dir,
                scheduler, resume_checkpoint=None, is_LoRA=False):
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0

    if resume_checkpoint:
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_training_loss = 0

        train_iterator = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch + 1}/{num_epochs}"
        )

        for batch in train_iterator:
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=inputs,
                attention_mask=masks,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_iterator.set_postfix({'Training Loss': loss.item()})
            epoch_training_loss += loss.item()

        avg_epoch_training_loss = epoch_training_loss / len(train_iterator)

        model.eval()

        epoch_validation_loss = 0
        total_loss = 0
        valid_iterator = tqdm(
            valid_loader,
            desc=f"Validation Epoch {epoch + 1}/{num_epochs}"
        )
        with torch.no_grad():
            for batch in valid_iterator:
                inputs = batch["input_ids"].to(device)
                masks = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=inputs,
                    attention_mask=masks,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss
                valid_iterator.set_postfix({'Validation Loss': loss.item()})
                epoch_validation_loss += loss.item()

        avg_epoch_validation_loss = epoch_validation_loss / len(valid_loader)

        print("saving")

        if is_LoRA:
            model.save_pretrained("./recipe_gpt2_LoRA")
            tokenizer.save_pretrained("./recipe_gpt2_LoRA")
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': avg_epoch_training_loss,
                'validation_loss': avg_epoch_validation_loss,
                'scheduler_state_dict': scheduler.state_dict(),
            }

            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt")

            if epoch > 0:
                os.remove(f"{checkpoint_dir}/checkpoint_epoch_{epoch - 1}.pt")

        print(f"Epoch: {epoch + 1}, Validation Loss: {total_loss / len(valid_loader)}")


def evaluate_model(model, test_loader, device):
    print(2)
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            active_tokens = labels != -100
            num_tokens = active_tokens.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":

    # загрузка и предобработка данных
    data = pd.read_csv('RecipeNLG_dataset.csv')
    data = data.drop(columns=["Unnamed: 0", "link", "source", "NER"])

    data["ingredients"] = data["ingredients"].apply(ast.literal_eval)
    data["directions"] = data["directions"].apply(ast.literal_eval)

    data["ingredients"] = data["ingredients"].apply(
        lambda x: [preprocess_ingredient(ingr) for ingr in x]
    )

    # выбор доступного устройства
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # загрузка модели для заголовков
    tokenizer_title = GPT2Tokenizer.from_pretrained('gpt2')

    if tokenizer_title.pad_token is None:
        tokenizer_title.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer_title.add_special_tokens(
        {"additional_special_tokens": ["<INGR_START>", "<INGR_END>", "<TITLE_START>", "<TITLE_END>", ]})

    model_title = GPT2LMHeadModel.from_pretrained('gpt2')
    model_title.resize_token_embeddings(len(tokenizer_title))
    model_title.to(device)

    # загрузка модели для рецептов
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<INGR_START>", "<INGR_END>", "<TITLE_START>",
                                                                "<TITLE_END>", "<RECIPE_START>", "<RECIPE_END>"]})
    config = GPT2Config.from_pretrained('gpt2')
    config.resid_pdrop = 0.4
    config.embd_pdrop = 0.4
    config.attn_pdrop = 0.4
    config.summary_first_dropout = 0.3

    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device)
    model.transformer.wte.requires_grad_(False)
    model.transformer.wpe.requires_grad_(False)

    # применение LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn"]
    )
    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))

    # подготовка данных для обучения модели заголовков
    titles = TitleDataset(data[:100000], tokenizer_title)

    train_size_title = int(0.7 * len(titles))
    valid_size_title = int(0.4 * (len(titles) - train_size_title))
    test_size_title = len(titles) - train_size_title - valid_size_title

    train_data_title, tmp_data_title = random_split(titles, [train_size_title, len(titles) - train_size_title])
    valid_data_title, test_data_title = random_split(tmp_data_title, [valid_size_title, test_size_title])

    batch_size = 6
    train_loader_title = DataLoader(train_data_title, batch_size=batch_size, shuffle=True)
    valid_loader_title = DataLoader(valid_data_title, batch_size=batch_size)
    test_loader_title = DataLoader(test_data_title, batch_size=batch_size)

    # подготовка данных для обучения модели рецептов
    recipes = RecipeDataset(data[:100000], tokenizer)

    train_size = int(0.7 * len(recipes))
    valid_size = int(0.4 * (len(recipes) - train_size))
    test_size = len(recipes) - train_size - valid_size

    train_data, tmp_data = random_split(recipes, [train_size, len(recipes) - train_size])
    valid_data, test_data = random_split(tmp_data, [valid_size, test_size])

    batch_size = 6
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # обучение модели заголовков
    num_epochs = 3
    optimizer = optim.Adam(model_title.parameters(), lr=3e-5)

    total_steps_title = len(train_loader_title) * num_epochs
    warmup_steps_title = int(0.1 * total_steps_title)

    scheduler_title = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps_title,
        num_training_steps=total_steps_title,
        num_cycles=0.5
    )

    train_model(model_title, num_epochs, optimizer, train_loader_title, valid_loader_title, batch_size, tokenizer_title,
                device, checkpoint_dir='./checkpoints_title', scheduler=scheduler_title)

    model_title.save_pretrained("./recipe_gpt2_title")
    tokenizer_title.save_pretrained("./recipe_gpt2_title")

    # обучение модели рецептов
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )

    train_model(model, num_epochs, optimizer, train_loader, valid_loader, batch_size, tokenizer, device,
                checkpoint_dir='./checkpoints', scheduler=scheduler, is_LoRA=True)

    model.save_pretrained("./recipe_gpt2_LoRA")
    tokenizer.save_pretrained("./recipe_gpt2_LoRA")

    # оценка работоспособности
    test_loss_title = evaluate_model(model_title, test_loader_title, device)
    test_loss = evaluate_model(model, test_loader, device)

    perplexity_title = math.exp(test_loss_title)
    print(f"Perplexity for titles: {perplexity_title:.2f}")
    perplexity = math.exp(test_loss)
    print(f"Perplexity for recipes: {perplexity:.2f}")





