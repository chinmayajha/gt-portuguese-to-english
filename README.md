# Portuguese to English Translation with Attention

This notebook implements a sequence-to-sequence (seq2seq) neural translation model to translate Portuguese sentences into English. The core innovation lies in the use of an **attention mechanism**, which improves translation quality by allowing the model to dynamically focus on relevant words in the input sequence.

## What It Does

- Translates Portuguese → English using a neural network
- Trains from scratch on a parallel corpus
- Uses attention to handle long or complex sentences more effectively

## How It Works

1. **Preprocessing**: Tokenizes and pads sentence pairs using Keras utilities.
2. **Model Architecture**:
   - **Encoder**: Bidirectional LSTM encodes the Portuguese input.
   - **Bahdanau Attention**: Computes alignment scores between decoder state and encoder outputs.
   - **Decoder**: LSTM uses attention context + previous outputs to generate English sentence.
3. **Training**: Custom loop with teacher forcing and gradient updates using `tf.GradientTape`.
4. **Inference**: Greedy decoding to translate new input sentences.

## What's Special

- **Attention Layer** is manually implemented to visualize alignment between input and output.
- Unlike fixed context vectors, attention allows the decoder to adaptively retrieve relevant encoder states at each time step.
- Entire architecture is built from scratch using low-level TensorFlow, making it ideal for understanding how attention works in NMT.

