# Restart the notebook (to flush the GPU memory)
from utils import restart_jupyter_notebook
#restart_jupyter_notebook()


# Import necessary packages and methods
from utils import *
import torch


# Default hyperparams, also defined in `utils.py` in class `Hyperparameters`
## !!! `model_name` attr must point to the location of the model weights !!!
## Weights can be downloaded from: https://llama.meta.com/llama-downloads/
hyperparams.model_name = "../../../../gemma-weights"  # <== Add model weight location here e.g. "/path/to/downloaded/llama/weights"
hyperparams.mixed_precision = "bf16"


# Init the model and accelerator wrapper
model = init_baseline_model(hyperparams).cuda()
model = model.to(torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
inputs = tokenizer(["Some random initial str ", "Another string ... "] * 32, return_tensors="pt", padding=True)

inputs['input_ids'] = inputs['input_ids'].cuda()
inputs['attention_mask'] = inputs['attention_mask'].cuda()


# Początek pomiaru czasu
start_time = time.time()

import pdb 
pdb.set_trace()
outputs = model.generate(
    **inputs,
    max_new_tokens=1000
)

# Koniec pomiaru czasu
end_time = time.time()

# Obliczamy czas trwania operacji
duration = end_time - start_time



print(duration)

# Decode the output tensor to text
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Display the generated text
for text in generated_texts:
    print(text)
    print("=" * 100)