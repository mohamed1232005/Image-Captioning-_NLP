# Multi-Model Image Captioning using Deep Learning

This project implements and compares multiple deep learning models for automatic image caption generation. The approach combines a pre-trained **ResNet152V2** CNN as a feature extractor and various decoders (LSTM, GRU, BART, GPT-2) to generate detailed captions for images. We evaluate the performance of these models using **BLEU** score and provide an interactive **Flask** web application for real-time image captioning.

## 📌 Project Overview

In this project, the goal is to develop a model that can automatically generate captions for images. We use the **Flickr8k** dataset, which contains 8,000 images with five human-written captions per image. To improve the model's performance and capability, we used modern deep learning techniques such as **ResNet152V2** for feature extraction and state-of-the-art decoders like **LSTM**, **GRU**, **BART**, and **GPT-2** for caption generation.

Key steps of this project include:
1. **Image Feature Extraction**: Using **ResNet152V2** to extract high-level visual features from images.
2. **Caption Generation**: Using **LSTM**, **GRU**, **BART**, and **GPT-2** models to generate captions based on the extracted features.
3. **Evaluation**: Evaluating the generated captions using **BLEU** score, a metric commonly used for assessing machine-generated text.
4. **Deployment**: Deploying the trained models via a **Flask API** to allow users to upload images and generate captions in real-time.

## 🗂️ Dataset

We used the **Flickr8k** dataset, which is available on Kaggle and consists of:
- 8,000 images, each with 5 captions written by humans.
- These images cover a wide variety of real-world scenes such as animals, people, landscapes, etc.

The dataset was preprocessed by:
- Converting the captions to lowercase.
- Removing punctuation and special characters.
- Tokenizing the captions using **TensorFlow’s Tokenizer**.
- Padding sequences to ensure uniform input size for the models.

Link to the dataset: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## 🧠 Model Architecture

### 📷 Feature Extraction (ResNet152V2)
- We used **ResNet152V2** as the image feature extractor. This model was pre-trained on the **ImageNet** dataset, allowing it to learn deep and complex representations of visual features from the image. We chose **ResNet152V2** due to its deeper architecture, which leads to more robust feature extraction compared to other shallower models like VGG16.
- The ResNet152V2 model is used without the fully connected layers (i.e., the classification head), so it outputs a 2048-dimensional vector as the image representation.
- The output features are passed to the decoder models (LSTM, GRU, BART, GPT-2).

### 🗣️ Decoders (Caption Generators)
For generating captions, we explored the following models:

1. **LSTM (Long Short-Term Memory)**:
   - LSTM is a type of Recurrent Neural Network (RNN) that is designed to learn long-term dependencies in sequence data.
   - In this model, we use the image features from ResNet152V2 as the initial input to the LSTM network, which sequentially generates the caption word by word.

2. **GRU (Gated Recurrent Unit)**:
   - GRU is another type of RNN similar to LSTM but with a simpler architecture.
   - It has fewer gates and parameters, which makes it faster to train while still being effective at sequence learning.

3. **BART (Bidirectional and Auto-Regressive Transformers)**:
   - BART is a modern transformer-based model that combines the benefits of both bidirectional and auto-regressive transformers.
   - It has been shown to be effective in many NLP tasks, including text generation. We use it as a decoder to generate more complex and fluent captions.

4. **GPT-2 (Generative Pre-trained Transformer 2)**:
   - GPT-2 is a state-of-the-art autoregressive language model. It uses a transformer architecture to predict the next word in a sequence based on the context of the previous words.
   - By leveraging GPT-2, we can generate highly fluent and coherent captions.

### 🧑‍💻 Training and Fine-tuning
- All models were trained using the same feature set extracted by **ResNet152V2**.
- We used **Adam optimizer** for training and employed **categorical cross-entropy** as the loss function.
- The training was performed with a batch size of 64 images and captions, with early stopping and learning rate reduction based on validation loss to prevent overfitting.

### 📊 Evaluation Metrics
We evaluated the models using the **BLEU** score, a precision-based metric that evaluates how many n-grams (i.e., sequences of n words) in the predicted captions match those in the reference captions. A higher BLEU score indicates that the model generated captions more similar to the ground truth.

Additionally, **Semantic Similarity** and **ROUGE** scores were considered, but BLEU was the primary metric for comparison.

### Evaluation Results

| Model  | BLEU Score   | Comments              |
|--------|--------------|-----------------------|
| **GPT-2** | 0.01722     | Lowest BLEU, but high fluency |
| **BART**  | 0.015113    | Similar to GPT-2, but slightly less fluent |
| **LSTM**  | 9.03e-155   | Very low BLEU, but captures semantics better |
| **GRU**   | 0.041       | Higher than LSTM, but lower than GPT-2 and BART |

## 🖥️ Deployment

The models were deployed using a **Flask** web API, which allows users to upload an image and receive captions in real-time. The Flask app includes:
- An image upload form for the user.
- Integration with the trained models to generate captions using the **BART** and **GPT-2** models.
- Display of BLEU score and generated captions for comparison.

### Usage:
1. **Train**: The models are first trained with the images and captions from the **Flickr8k** dataset.
2. **Web Interface**: The user uploads an image, and the web interface generates captions using the trained models (e.g., BART and GPT-2).
3. **BLEU Scores**: For each caption, the BLEU score is calculated, and users can compare captions generated by different models.

## 🧰 Requirements

The project requires the following libraries:

- TensorFlow
- Keras
- PyTorch
- Hugging Face Transformers
- Flask
- NLTK
- OpenCV
- PIL
- matplotlib
- tqdm

To install all dependencies, run:

```bash
pip install -r requirements.txt
