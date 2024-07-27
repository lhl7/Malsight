from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq


"""
nohup python ft1_encoder_only_codet5p_770m.py > ../logs/ft1.log 2>&1 &
"""

total = 93809
train = 60000
ds = Dataset.from_json('../datasets/code_sum_with_anno_longer_label_93809.json').train_test_split(total-train, seed=64)

model_path = '../models/hf/codet5p-770m'
ft_model_path = '../checkpoints/diff_ratio/3:4/p1/checkpoint-625'

tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                          trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path,
                                              trust_remote_code=True)

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# freeze the decoder
model.decoder.requires_grad_(False)

print('load model successfully')


def process(sample):
    codes = [' '.join(code) for code in sample['code_tokens']]
    annos = [' '.join(anno) for anno in sample['api_call_anno']]
    targets = [' '.join(s) + ', ' + l[0].lower() + l[1:] for s, l in zip(sample['docstring_tokens'], sample['longer_label'])]
    
    
    inputs = tokenizer(codes, 
                       text_pair=annos,
                       max_length=400, 
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

tokenized_dataset['train'] = tokenized_dataset['train'].remove_columns(['api_call', 'api_call_anno', 'code'])

save_dir = '../checkpoints/ft1'

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
    fp16=True,
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