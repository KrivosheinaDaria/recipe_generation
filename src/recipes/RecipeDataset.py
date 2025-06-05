from torch.utils.data import Dataset

class RecipeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1000):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.ingr_start_id = tokenizer.convert_tokens_to_ids("<INGR_START>")
        self.ingr_end_id = tokenizer.convert_tokens_to_ids("<INGR_END>")
        self.title_start_id = tokenizer.convert_tokens_to_ids("<TITLE_START>")
        self.title_end_id = tokenizer.convert_tokens_to_ids("<TITLE_END>")
        self.recipe_start_id = tokenizer.convert_tokens_to_ids("<RECIPE_START>")
        self.recipe_end_id = tokenizer.convert_tokens_to_ids("<RECIPE_END>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        ingredients = ', '.join(row['ingredients'])
        title = row['title']
        recipe_list = row['directions']
        recipe = ''

        for i in range(len(recipe_list)):
            recipe = recipe + recipe_list[i] + ' '
        text = (
            f"<INGR_START> {ingredients} <INGR_END>\n"
            f"\n<TITLE_START> {title} <TITLE_END>\n"
            f"<RECIPE_START> {recipe} <RECIPE_END>"
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
            recipe_start_pos = list(input_ids).index(self.recipe_start_id)
            recipe_end_pos = list(input_ids).index(self.recipe_end_id)
        except ValueError:
            labels[:] = -100
            recipe_start_pos = 0
            recipe_end_pos = 0

        labels[:recipe_start_pos] = -100
        labels[recipe_end_pos:] = -100

        special_tokens_mask = (
                (input_ids == self.ingr_start_id) |
                (input_ids == self.ingr_end_id) |
                (input_ids == self.recipe_start_id) |
                (input_ids == self.recipe_end_id) |
                (input_ids == self.title_start_id) |
                (input_ids == self.title_end_id)
        )
        labels[special_tokens_mask] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }

