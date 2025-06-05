from torch.utils.data import Dataset

class TitleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1000):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.title_start_id = tokenizer.convert_tokens_to_ids("<TITLE_START>")
        self.title_end_id = tokenizer.convert_tokens_to_ids("<TITLE_END>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        ingredients = ', '.join(row['ingredients'])
        title = row['title']

        text = (
            f"<INGR_START> {ingredients} <INGR_END>\n"
            f"<TITLE_START> {title} <TITLE_END>"
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()

        try:
            title_start_pos = list(input_ids).index(self.title_start_id)
            title_end_pos = list(input_ids).index(self.title_end_id)
        except ValueError:
            labels[:] = -100
            title_start_pos = 0
            title_end_pos = 0

        labels[:title_start_pos] = -100
        labels[title_end_pos + 1:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }



