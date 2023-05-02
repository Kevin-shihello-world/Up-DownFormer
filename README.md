# Up-DownFormer
Up-DownFormer: This kind of transformer architecture is mostly a newly decided GNN  decided in this work, And I've tested this kind of gene and on normal GNN test and get superior result and  thishe whole new transformer architecture on a NLP task and it got comparable result as the formal all self-attention ones with much lower computation
In order to test it on GLUE just make sure you've downloaded pre-trained model from Huggingface or your cloud server provider in folder model_ and Glue dataset in folder dataset_ and input the follow to the terminal:
python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name qnli \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/qnli/
p.s:For new comer like who I am(laugh):the "run_glue_no_trainer.py" should be the path of run_glue_no_trainer document in your environment, if you put it in a new folder the terminal won't go to search it from the whole environment.
