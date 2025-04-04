# Up-DownFormer

Up-DownFormer: This kind of transformer architecture is mostly a newly decided GNN  decided in this work, And I've tested this kind of gene and on normal GNN test and get superior result and  thishe whole new transformer architecture on a NLP task and it got comparable result as the formal all self-attention ones with much lower computation
In order to test it on GLUE just make sure you've downloaded pre-trained model from Huggingface or your cloud server provider in folder model_ and Glue dataset in folder dataset_ and input the follow to the terminal:
(use QNLI as an example)
```
python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name qnli \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/qnli/
```
THe result as I tested isaccuracy of 0.9051803
p.s:For new comers:the "run_glue_no_trainer.py" should be the path of run_glue_no_trainer document in your environment, if you put it in a new folder the terminal won't go to search it from the whole environment.

And in the latest UP-DOWNformer2.0.py file contains the newly desided 2.0 version of UP-DOWNformer
# UP-DOWNformer 2.0

UP-DOWNformer 2.0 is an innovative transformer architecture that introduces cross-layer attention mechanisms and scientifically utilizes the sparsity of self-attention matrices. This document provides an overview of its architecture, technical principles, performance advantages, and seamless integration with HuggingFace Transformers.

## Key Features

### Cross-Layer Attention Mechanism
UP-DOWNformer 2.0 enhances the cross-layer attention mechanism with dynamic re-weighting and precise token selection strategies for non-initial layers, enabling more accurate predictions while reducing computational overhead.

### Sparse Attention Matrix Utilization
UP-DOWNformer 2.0 improves upon version 1.0 by implementing optimized sparse attention patterns for deeper layers, achieving greater computational efficiency while maintaining prediction accuracy.

### HuggingFace Integration
UP-DOWNformer 2.0 is designed for seamless integration with the HuggingFace Transformers library, making it easy to incorporate into existing NLP pipelines.

## Technical Details

The architecture consists of:
- **Up Projection**: Compresses information from the attention matrix in the first layer to pass to the following layers(I call it compressed attention information),and in the following layers to the first one, get information from hidden states to the selcted attention interact tokens and their weights to the compressed attention information to  pass to next layer
- **Down Projection**: Use the compressed attention information to help compute the approximated attention matrix in the next layer;So the UP-DOWN strategy let the attention information flow attention information with the data processing section like MLP and norms to bring more information for the layers followed the first layer to generate its approximated attention matrix.
- **Gated Activation**: Controls information flow between layers

## Performance Benefits
- Reduced memory footprint
- Faster inference speed
- Improved gradient flow
- Better long-range dependency modeling
