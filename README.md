# Tech_challenge_3

## Tratamento e preparação dos dados 

Para o tratamento e preparação da base obtamos por fazer a limpeza de campos vazios e nulos, juntamente com a remoção de caracteres especiais e HTML, com isso retiramos da base as colunas que não utilizariamos para o fine tunning elas são as 'target_ind', 'target_rel' e 'uid'.
Com a base limpa nos optamos por quebrar a mesma em pedaços para melhorar a performance na hora de preparar o modelo pra tokenizar


## Fine Tunning
utilizamos o modelo ```unsloth/Llama-3.2-1B-bnb-4bit``` com os parametros: 

### BATCH - Otimizado para A100 sem estourar RAM
per_device_train_batch_size=24,  # Máximo seguro para A100

per_device_eval_batch_size=24,

gradient_accumulation_steps=1,  # Sem acumulação = mais rápido

### LEARNING
learning_rate=5e-4,

lr_scheduler_type="cosine",

warmup_ratio=0.03,

### PRECISÃO
bf16=True,

bf16_full_eval=True,

### LOGGING MÍNIMO
logging_steps=100,

logging_first_step=True,

### SEM SALVAMENTO DURANTE TREINO
save_strategy="no",

save_steps=999999,

### AVALIAÇÃO MÍNIMA
eval_strategy="steps",

eval_steps=500,  # Apenas 4 avaliações

### STEPS
num_train_epochs=1,

max_steps=2000,

### OTIMIZAÇÕES
optim="adamw_torch_fused",

gradient_checkpointing=True,

### DATALOADER RÁPIDO
dataloader_num_workers=2,  # 2 workers = balanço velocidade/RAM

dataloader_pin_memory=True,

### SEM EXTRAS
report_to="none",

load_best_model_at_end=False,

disable_tqdm=False,
