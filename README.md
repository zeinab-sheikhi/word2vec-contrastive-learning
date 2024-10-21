# Word2Vec Embeddings with Contrastive Learning on IMDB Dataset

## Overview
This project implements Word2Vec embeddings using the IMDB dataset with contrastive learning algorithms. The aim is to train a model to maximize the similarity between a word and its positive context while minimizing similarity with negative contexts.

## Data Preprocessing
- The `DataLoaderWrapper` class loads and preprocesses the IMDB dataset.
- Uses BertTokenizer for tokenization, then prepares data loaders for training and validation.
- Extracts context words for each word based on a specified radius.
- Handles data shuffling, tokenization, and dataset splitting.

## Model Definition
- The `Word2Vec` class has two embedding layers for target and context words.
- Computes similarity scores between target words and both positive and negative contexts.
- `contrastive_loss` method calculates the loss using log loss for positive and negative contexts.
- The `Trainer` class manages training and validation using Adam optimizer and a learning rate scheduler.

## Ablation Study
| Hyperparams | R | K | d | B | E | Number of Samples |
|-------------|---|---|---|---|---|-------------------|
| HP1         | 5 | 10| 100| 16| 5 | 5000             |
| HP2         | 5 | 10| 300| 16| 5 | 5000             |
| HP3         | 3 | 5 | 100| 16| 5 | 5000             |
| HP4         | 3 | 5 | 100| 16| 5 | 10000            |
| HP5         | 4 | 5 | 100| 16| 5 | 50000            |

Results show that the best parameters for training are **R = 4, K = 5, d = 100**.

## Evaluation of Word2Vec Embeddings
The `WordEmbeddingEvaluator` class includes methods for:
- `get_word_embedding`: Retrieves embeddings for words.
- `evaluate_similarity`: Calculates cosine similarities between words.
- `analogy`: Finds words that complete analogies.
- `nearest_neighbors`: Identifies top-k similar words.
- `visualize_embeddings`: Uses t-SNE for 2D visualization of embeddings.

## Results
### Validation Results
| Hyperparams | Loss   | Accuracy |
|-------------|--------|----------|
| HP1         | 0.0886 | 0.82333  |
| HP2         | 0.1103 | 0.8234   |
| HP3         | 0.0868 | 0.8391   |
| HP4         | 0.0424 | 0.8761   |
| HP5         | 0.0113 | 0.9282   |

The model achieved its best performance with **R = 4, K = 5, d = 100** using the full dataset.

### Intrinsic Evaluation
The model learned meaningful representations for sentiment analysis, grouping positive and negative words separately in a t-SNE plot.

## Using Pretrained Word2Vec Embeddings in ConvModel
The `Conv1dClassifier` class uses pretrained Word2Vec embeddings for text classification:
- Supports both frozen and trainable embeddings.
- Allows using random or pretrained word embeddings based on task needs.
- Results indicate that the model with randomly initialized embeddings performed better than the one using pretrained Word2Vec embeddings.

### Training Results
| Epoch | Train Loss | Valid Loss | Train Accuracy | Valid Accuracy |
|-------|------------|------------|----------------|----------------|
| 1     | 0.71       | 0.65       | 54.65          | 58.70          |
| 5     | 0.42       | 0.44       | 80.83          | 78.70          |

- The randomly initialized embeddings achieved better results likely due to their ability to adapt more to the specific classification task.
