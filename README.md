# Text Embedding Model

A Rust application that generates text embeddings using FastText, designed for natural language processing tasks.

## Overview

This project implements a text embedding system that:
- Tokenizes input text using WordPiece tokenization
- Generates embeddings using a FastText model
- Calculates similarity between text embeddings
- Serializes and saves embeddings to JSON format

The embeddings are 100-dimensional vectors that capture semantic meaning of text, which can be used for various NLP tasks such as semantic similarity, classification, and clustering.

## Features

- **Custom Tokenization**: Uses WordPiece tokenization with normalization (lowercase, accent stripping)
- **FastText Integration**: Leverages the FastText library for high-quality word embeddings
- **Model Persistence**: Automatically saves and loads the trained model
- **Embedding Serialization**: Saves generated embeddings to JSON for easy use in other applications
- **Similarity Calculation**: Built-in cosine similarity function for comparing embeddings

## Installation

### Prerequisites

- Rust (latest stable version)
- Cargo package manager

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/embedddingmodel.git
   cd embedddingmodel
   ```

2. Build the project:
   ```bash
   cargo build --release
   ```

## Usage

### Basic Usage

Run the application with:

```bash
cargo run
```

This will:
1. Set up the embedding model
2. Load an existing FastText model (or train a new one if none exists)
3. Generate embeddings for sample sentences
4. Calculate similarity between embeddings
5. Save the embeddings to `embeddings.json`

### Using in Your Own Code

```
// Example code for using the embedding model
fn example_usage() -> Result<()> {
    // Create a tokenizer
    let tokenizer = create_tokenizer()?;

    // Load or train the embedding model
    let embedding_model = train_or_load_model()?;

    // Generate embedding for text
    let text = "Your text here";
    let embedding = embed_text(&tokenizer, &embedding_model, text)?;

    // Calculate similarity between two embeddings
    let similarity = cosine_similarity(
        &embedding.embedding,
        &another_embedding.embedding
    );

    println!("Similarity: {}", similarity);

    Ok(())
}
```

## Project Structure

- `main.rs`: Contains all the code for the application
- `Cargo.toml`: Project dependencies and configuration
- `fasttext_model.bin`: The trained FastText model (generated on first run)
- `embeddings.json`: Output file containing generated embeddings

## Dependencies

- `tokenizers`: For text tokenization (from Hugging Face)
- `fasttext`: Rust bindings for the FastText library
- `anyhow`: For error handling
- `serde` and `serde_json`: For JSON serialization
- `ndarray`: For numerical operations

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
