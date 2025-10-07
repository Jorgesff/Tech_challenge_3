# Tech_challenge_3

## Tratamento e preparação dos dados 

Para o tratamento e preparação da base obtamos por fazer a limpeza de campos vazios e nulos, juntamente com a remoção de caracteres especiais e HTML, com isso retiramos da base as colunas que não utilizariamos para o fine tunning elas são as 'target_ind', 'target_rel' e 'uid'.
Com a base limpa nos optamos por quebrar a mesma em pedaços para melhorar a performance na hora de preparar o modelo pra tokenizar


## Fine Tunning

Optamos por utilizar o modelo `unsloth/llama-3-8b-bnb-4bit` com os seguintes parametros:

per_device_train_batch_size=1

gradient_accumulation_steps=8

learning_rate=2e-4      

fp16=True

logging_steps=50

save_strategy="steps"

save_steps=1000

num_train_epochs=1

max_steps=4000

save_total_limit=5

report_to="none"
