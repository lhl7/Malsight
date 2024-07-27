from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq


"""
nohup python ft0_codet5p_770m.py > ../logs/ft0.log 2>&1 &
"""

total = 89609
train = 80000

ds = Dataset.from_json('../datasets/src_code.json').train_test_split(total-train, seed=21)

model_path = '../models/hf/codet5p-770m'
ft_model_path = '../models/hf/codet5p-770m'

tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                          trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path,
                                              trust_remote_code=True)

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

def process(sample):
    codes = [' '.join(code) for code in sample['code_tokens']]
    targets = [' '.join(s) for s in sample['docstring_tokens']]
    
    inputs = tokenizer(codes, 
                       max_length=320, 
                       padding='max_length', 
                       truncation=True)
    labels = tokenizer(targets,
                       max_length=120,
                       padding='max_length',
                       truncation=True)
    
    inputs['labels'] = labels['input_ids'].copy()
    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in inputs["labels"]
    ]
    return inputs

tokenized_dataset = ds.map(
    process, 
    batched=True,
    num_proc=64,
    load_from_cache_file=False,
)

tokenized_dataset['train'] = tokenized_dataset['train'].remove_columns(['code_tokens', 
                                                                        'docstring_tokens',
                                                                        'id',
                                                                        'fun_name',
                                                                        'repo',
                                                                        'starting',
                                                                        'partition',
                                                                        ])

save_dir = '../checkpoints/ft0'

training_args = TrainingArguments(

    output_dir=save_dir,
    overwrite_output_dir=False,

    do_train=True,
    save_strategy='epoch',

    num_train_epochs=1,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,

    learning_rate=5e-5,
    weight_decay=0.05,
    warmup_steps=200,
    
    logging_first_step=True,
    logging_steps=10,

    dataloader_drop_last=True,
    # dataloader_num_workers=4,

    # local_rank=-1,
    # deepspeed=None,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)

trainer.train()

# final_checkpoint_dir = os.path.join(save_dir, "final_checkpoint")
# model.save_pretrained(final_checkpoint_dir)

