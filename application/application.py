from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import tkinter as tk
from transformers import GPT2LMHeadModel
import torch
from transformers import GPT2Tokenizer
from peft import AutoPeftModelForCausalLM
import re

class Application(Tk):
    def __init__(self):
        super().__init__()
        self.title("~RECIPE GENERATION~")
        self.geometry("1200x1050")
        self.configure(bg="#fffff0")
        self.style = ttk.Style()
        self.create_widgets()
        self.load_models()

    def load_models(self):
        print("Загрузка модели для заголовков...")
        self.model_title = GPT2LMHeadModel.from_pretrained("recipe_gpt2_title")
        self.tokenizer_title = GPT2Tokenizer.from_pretrained("recipe_gpt2_title")
        self.model_title.eval()

        print("Загрузка модели для рецептов...")
        self.tokenizer_recipe = GPT2Tokenizer.from_pretrained("recipe_gpt2_LoRA")
        self.model_recipe = AutoPeftModelForCausalLM.from_pretrained("recipe_gpt2_LoRA")
        self.model_recipe.resize_token_embeddings(len(self.tokenizer_recipe))
        self.model_recipe.base_model.eval()

        print("Модели успешно загружены!")

    def create_widgets(self):
        self.style.configure(
            'Custom.Label',
            background='#4C9141',
            foreground='#fffff0',
            relief='solid',
            font=("Georgia", 10),
        )
        self.frame1 = ttk.Frame(self, width=1100, height=400, style="Custom.Label")
        self.frame1.pack(padx=10, pady=10)
        self.label1 = ttk.Label(self.frame1, text="enter the ingredients (separate them with commas)",
                                style="Custom.Label", width=100)
        self.label1.place(relx=0.02, rely=0.0005, anchor="nw")

        self.ScrolledText = ScrolledText(self.frame1, width=40, font=("Georgia", 12), bg="#fffff0",
                                         fg="#000000",  wrap=tk.WORD, insertbackground="black")
        self.ScrolledText.place(relx=0.02, rely=0.1, anchor="nw", relwidth=0.7, relheight=0.8)
        self.ScrolledText.vbar.config(
            width=20,
            borderwidth=2,
            cursor="hand2",
            relief="raised",
            bd=5,
            highlightthickness=2,
            highlightcolor="#d9d9d9",
            highlightbackground="#737373"
        )

        self.button = tk.Button(
            self.frame1,
            text="COOK!",
            bg="#1B5E20",
            cursor="hand2",
            fg="white",
            font=("Georgia", 12, "bold"),
            relief="raised",
            bd=3,
            activebackground="#2E7D32",
            activeforeground="white",
            highlightthickness=2,
            highlightcolor="#4CAF50",
            highlightbackground="#4CAF50"
        )
        self.button.place(
            relx=0.75,
            rely=0.2,
            anchor="nw",
            width=240,
            height=150
        )

        self.frame2 = ttk.Frame(self, width=1100, height=600, style="Custom.Label")
        self.frame2.pack(padx=20)
        self.ScrolledText2 = ScrolledText(self.frame2, width=20, font=("Georgia", 12), bg="#fffff0",
                                         fg="#000000", wrap=tk.WORD, insertbackground="black")
        self.ScrolledText2.place(relx=0.02, rely=0.05, anchor="nw", relwidth=0.95, relheight=0.9)
        self.ScrolledText2.configure(state='disabled')
        self.ScrolledText2.vbar.config(
            width=20,
            borderwidth=2,
            cursor="hand2",
            relief="raised",
            bd=5,
            highlightthickness=2,
            highlightcolor="#d9d9d9",
            highlightbackground="#737373"
        )
        self.button.config(command=self.generate_and_display)

    def preprocess_ingredient(self, ingredient):
        ingredient = re.sub(r"""[^a-zA-Z0-9\s/\.\(\),-]""", "", ingredient, flags=re.VERBOSE)
        ingredient = re.sub(r"\s+", " ", ingredient).strip().lower()
        replacements = {
            r"\bc\.?\b": "cup",
            r"\bpkg\.?\b": "package",
            r"\bml\.?\b": "milliliter",
            r"\bgr\.?\b": "gram",
            r"\btbsp\.?\b": "tablespoon",
            r"\btsp\.?\b": "teaspoon",
            r"\bkg\.?\b": "kilogram",
            r"\boz\.?\b": "ounce",
            r"\blb\.?\b": "pound"
        }
        for pattern, replacement in replacements.items():
            ingredient = re.sub(pattern, replacement, ingredient)
        return ingredient

    def remove_unfinished_sentence(self, text):
        if not text:
            return text
        if re.search(r'[.!?]$', text):
            return text
        match = re.search(r'.*[.!?]', text)
        if match:
            return match.group(0)
        else:
            return text

    def clean_generated_text(self, text):
        text = text.replace("<INGR_START>", "+").replace("<INGR_END>", "+").replace("<TITLE_START>", "+").replace(
            "<TITLE_END>", "+")
        text = text.replace("<RECIPE_START>", "+").replace("<RECIPE_END>", "+")
        text = text.replace('!', '.')
        text = text.replace('?', '.')
        text = re.sub(r'([.,;:?])\1{1,}', r'\1', text)
        text = re.sub(r'\.{2,}', '.', text)

        text = text.split('+')

        cleaned_text = list(map(lambda a: re.sub(r'[^a-zA-Z0-9.,!°?+;:\'"\(\)\[\]\{\}\-%\s\n]', '', a), text))

        cleaned_text = list(map(lambda a: re.sub(r'\([^)]*\)', '', a), cleaned_text))
        cleaned_text = list(map(lambda a: re.sub(r'\(', '', a), cleaned_text))
        cleaned_text = list(map(lambda a: re.sub(r'\)', '', a), cleaned_text))

        cleaned_text = list(map(lambda a: re.sub(r'\n+', '\n', a), cleaned_text))

        cleaned_text = list(map(lambda a: re.sub(r' +', ' ', a), cleaned_text))

        cleaned_text = list(map(lambda a: re.sub(r'\s+([.,!?;:)\]})])', r'\1', a), cleaned_text))

        cleaned_text = list(map(lambda a: re.sub(r'([.,!?;:])(?=[^\s\d])', r'\1 ', a), cleaned_text))

        cleaned_text = list(map(lambda a: a.strip(), cleaned_text))

        cleaned_text = list(filter(lambda a: a != '', cleaned_text))
        cleaned_text = list(map(lambda a: a.replace("°", "°F"), cleaned_text))

        return cleaned_text

    def generate_and_display(self):

        ingredients = self.ScrolledText.get("1.0", tk.END).strip()
        if not ingredients:
            return
        ingredients = self.preprocess_ingredient(ingredients)
        print(ingredients)

        self.button.config(text="thinking...", state="disabled")
        self.update()

        try:
            input_str = f"<INGR_START> {ingredients} <INGR_END> <TITLE_START>"
            input_ids_title = self.tokenizer_title.encode(input_str, return_tensors='pt')

            generated_title = self.model_title.generate(
                input_ids_title,
                max_length=300,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                eos_token_id=self.tokenizer_title.convert_tokens_to_ids("<TITLE_END>")
            )

            generated_title_id = generated_title.tolist()[0].index(self.tokenizer_title.convert_tokens_to_ids("<TITLE_START>"))
            generated_title = torch.tensor(generated_title.tolist()[0][generated_title_id + 1:])
            decoded_title = self.tokenizer_title.decode(generated_title, skip_special_tokens=False)

            input_str += decoded_title + " <RECIPE_START>"
            input_ids_recipe = self.tokenizer_recipe.encode(input_str, return_tensors='pt')
            attention_mask_recipe = (input_ids_recipe != self.tokenizer_recipe.pad_token_id).long()

            generated_recipe = self.model_recipe.base_model.generate(
                input_ids_recipe,
                max_length=300,
                min_length=100,
                attention_mask=attention_mask_recipe,
                pad_token_id=self.tokenizer_recipe.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.6,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
                eos_token_id=self.tokenizer_recipe.eos_token_id,
                bad_words_ids=[[self.tokenizer_recipe.encode(char)[0] for char in "[]{}<>"]],
                typical_p=0.90
            )
            decoded_recipe = self.tokenizer_recipe.decode(generated_recipe[0], skip_special_tokens=False)
            cleaned_recipe = self.clean_generated_text(decoded_recipe)
            cleaned_recipe[2] = cleaned_recipe[2].split('Ingredients')
            cleaned_recipe[2][0] = self.remove_unfinished_sentence(cleaned_recipe[2][0])

            self.ScrolledText2.config(state='normal')
            self.ScrolledText2.delete(1.0, tk.END)

            self.ScrolledText2.tag_configure("bold", font=("Georgia", 14, "bold"))
            self.ScrolledText2.tag_configure("normal", font=("Georgia", 12))

            self.ScrolledText2.insert(tk.END, "«" + cleaned_recipe[1] + "»\n\n", "bold")

            ingredients_list = cleaned_recipe[0].replace(', ', ',')
            ingredients_list = ingredients_list.split(',')
            formatted_ingredients = []
            for item in ingredients_list:
                item = item.strip()
                if item:
                    formatted_ingredients.append(f"• {item}")

            self.ScrolledText2.insert(tk.END, "Ingredients\n", "bold")
            self.ScrolledText2.insert(tk.END, "\n".join(formatted_ingredients) + "\n\n", "normal")

            self.ScrolledText2.insert(tk.END, "Recipe\n", "bold")
            recipe_text = cleaned_recipe[2][0] if len(cleaned_recipe) > 2 else 'Could not to come up with a recipe:('
            self.ScrolledText2.insert(tk.END, recipe_text, "normal")


        except Exception as e:
            self.ScrolledText2.config(state='normal')
            self.ScrolledText2.delete(1.0, tk.END)
            self.ScrolledText2.insert(tk.END, f"Error generation: {str(e)}", "normal")
        finally:
            self.ScrolledText2.config(state='disabled')
            self.button.config(text="COOK!", state="normal")


if __name__ == "__main__":
    app = Application()
    app.mainloop()
