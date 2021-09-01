# Decomposition-Table-Reasoning
Code for EMNLP2021 paper "Exploring Decomposition for Table-based Fact Verification"

## Pseudo-Decomposition Dataset
Available in ./Data, including: 
- train_decomp_type_detect_lm.json: training data for decomposition type detection
- train_decomp_generation_lm.json: training data for decomposition generation

## Decomposition Type Detection 
- Fine-tune [BERT](https://arxiv.org/pdf/1810.04805.pdf) model for decomposition type detection (5-way classification). 
- Predict decomposition type for all samples in the [TabFact](https://github.com/wenhuchen/Table-Fact-Checking) dataset.

## Decomposition Generation
- Fine-tune [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) model for decomposition generation.
- Generate decomposition for all samples in the TabFact dataset.

## Subproblem Solver
- After getting the decomposed sub-problems for samples in the TabFact, we solve them to obtain corresponding answers.
- How to verify the decomposed sub-statements?
  
  We used the TAPAS model fine-tuned on the TABFACT (Chen et al., 2020) for sub-statement verification, and the model is available [here](https://huggingface.co/google/tapas-large-finetuned-tabfact).
  
- How to answer the decomposed sub-questions?

  We used the TAPAS model fine-tuned on the WikiTableQuestions (Pasupat and Liang, 2015) for sub-question answering, and the model is available [here](https://huggingface.co/google/tapas-large-finetuned-wtq).

## Combine Intermediate Evidence for Fact Verification
- Data
  
  ```
  cd ./Data/TabFact_data
  tar -zxvf table.tar.gz
  ```
- Download model checkpoint [here](https://www.dropbox.com/sh/pmjhlqd51msiy3h/AACsfTpzDwDbTvCCIo7Vqx3-a?dl=0), and put them in the ./ckpt folder.
  
- Re-train the model

  If you would like to re-train the verification model:
  ```
  cd ./Code
  python run.py --do_train --do_eval --tune_tapas_after10k --load_tapas_model ../ckpt/base.pt --data_dir ../Data/TabFact_data
  ```
- Test the model 

  Or you can evaluate the model using the provided checkpoint:
  ```
  cd ./Code
  python run.py --do_eval --do_test --do_simple_test --do_complex_test --do_small_test --tune_tapas_after10k --load_model ../ckpt/model.pt --data_dir ../Data/TabFact_data
  ```
  
## Contact
For any questions, please send email to the author. 
