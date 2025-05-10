# DistilBERT for Named Entity Recognition (NER) on CoNLL-2003

This repository presents a complete pipeline for fine-tuning a transformer-based model—DistilBERT—for the task of Named Entity Recognition (NER) using the CoNLL-2003 dataset. The project is implemented in PyTorch and HuggingFace Transformers, and all code, experiments, and results are documented in the `distilbert-ner-conll2003.ipynb` notebook.

## Project Overview

The main objective is to train a model that can accurately identify and classify named entities such as persons, organizations, locations, and miscellaneous entities within text. The workflow begins with preprocessing and tokenizing the CoNLL-2003 dataset, followed by aligning the entity labels with the tokenized inputs. DistilBERT, a lightweight and efficient transformer, is then fine-tuned for token classification. For efficiency, the base model's parameters are frozen, and only the classification head is trained. Both a custom PyTorch training loop and the HuggingFace Trainer API are used to train and evaluate the model, providing a comparison of hands-on and high-level approaches.

## Dataset and Model

The CoNLL-2003 dataset is a widely used benchmark for NER, containing annotated sentences with entity tags for each word. DistilBERT, a distilled version of BERT, serves as the backbone for this project, offering a balance between performance and computational efficiency. The model is adapted for token classification, making it suitable for sequence labeling tasks like NER.

## Training and Evaluation

After preparing the data, the model is trained using both a custom loop and the HuggingFace Trainer. The custom loop provides transparency and control over the training process, while the Trainer streamlines experimentation and hyperparameter tuning. Evaluation is performed on both validation and test sets, with accuracy as the primary metric. The best-performing models are saved and can be easily loaded for inference or further experimentation.

## Training Details

- **Optimizer:** AdamW
- **Learning Rate:** 5e-5
- **Batch Size:** 16
- **Epochs:** 3
- **Max Sequence Length:** 128 tokens
- **Loss Function:** Cross-entropy (applied only to non-padding tokens)
- **Trainable Parameters:** Only the classification head (DistilBERT base is frozen)
- **Early Stopping:** Based on validation accuracy
- **Frameworks:** PyTorch (custom loop) and HuggingFace Trainer
- **Device:** GPU or CPU

## Results

The fine-tuned DistilBERT model achieves strong performance on the NER task. Using the custom PyTorch training loop, the model reaches approximately 92% accuracy on both validation and test sets. When trained with the HuggingFace Trainer, the model achieves an accuracy of about 95% on the validation set. These results demonstrate the effectiveness of transformer-based models for sequence labeling and highlight the benefits of leveraging both low-level and high-level training workflows.

- **Validation Accuracy (PyTorch loop):** ~0.92
- **Test Accuracy (PyTorch loop):** ~0.92
- **Validation Accuracy (HuggingFace Trainer):** ~0.95

The fine-tuned model and detailed results are available on HuggingFace:
- [Fine-tuned Model](https://huggingface.co/aren-golazizian/distilbert-ner-finetuned-conll2003)
- [Results](https://huggingface.co/aren-golazizian/results)

## Applications

Named Entity Recognition is a foundational task in natural language processing with broad applications. The techniques and models developed in this project can be applied to information extraction for business intelligence, news analytics, and search engines; conversational AI for chatbots and virtual assistants; and content analysis for social media, legal, and medical documents.
