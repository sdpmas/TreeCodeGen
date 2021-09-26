# !/bin/bash
set -e
seed=0
freq=3

vocab='data/conala/vocab.bin'
train_file='data/conala/train.bin'
test_file='data/conala/test.bin'
dev_file="data/conala/dev.bin"
dropout=0.1
hidden_size=256
embed_size=256
action_embed_size=256
field_embed_size=64
type_embed_size=64
lr=1e-4
lr_decay=0.5
batch_size=64
max_epoch=70
beam_size=30
lr_decay_after_epoch=15
model_name=retdistsmpl.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).seed${seed}


mkdir -p logs/conala


python exp.py \
    --seed ${seed} \
    --mode train \
    --batch_size ${batch_size} \
    --evaluator conala_evaluator \
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --test_file ${test_file} \
    --vocab ${vocab} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --max_epoch ${max_epoch} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to saved_models/abc



