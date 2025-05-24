use anyhow::{Result, anyhow};
use fasttext::FastText;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::{NFD, Lowercase, StripAccents, Sequence};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use serde::{Serialize, Deserialize};

// Define a constant for the embedding dimension
const EMBEDDING_DIMENSION: usize = 100;

// Structure to hold our embeddings for serialization
#[derive(Serialize, Deserialize)]
struct TextEmbedding {
    text: String,
    embedding: Vec<f32>,
}

fn main() -> Result<()> {
    println!("Setting up embedding model...");

    // Create a simple tokenizer
    let tokenizer = create_tokenizer()?;

    // Train a TF-IDF model on sample data (or load a pre-existing model)
    // For a real application, you'd want to train on a larger corpus
    let embedding_model = train_or_load_model()?;

    // Example text to generate embeddings for
    let sentences = vec![
        "This is a sample sentence",
        "Another example text for embedding",
    ];

    println!("Generating embeddings for sample sentences...");

    // Generate and store embeddings
    let mut embeddings = Vec::new();
    for sentence in &sentences {
        let embedding = embed_text(&tokenizer, &embedding_model, sentence)?;
        embeddings.push(embedding);

        // Save this embedding
        let text_embedding = TextEmbedding {
            text: sentence.to_string(),
            embedding: embeddings.last().unwrap().embedding.clone(),
        };

        // Print embedding details
        println!(
            "Generated embedding for: '{}' with {} dimensions",
            sentence,
            text_embedding.embedding.len()
        );
    }

    // Demo: Calculate similarity between embeddings if we have at least 2
    if embeddings.len() >= 2 {
        let similarity = cosine_similarity(
            &embeddings[0].embedding,
            &embeddings[1].embedding
        );
        println!("Cosine similarity between the two embeddings: {:.6}", similarity);
    }

    println!("Text embedding completed successfully!");

    // Save embeddings to a file as JSON
    save_embeddings_to_file(&embeddings, "embeddings.json")?;
    println!("Embeddings saved to embeddings.json");

    Ok(())
}

// Create a simple tokenizer
fn create_tokenizer() -> Result<Tokenizer> {
    // Create a basic vocabulary with common words and the [UNK] token
    let vocab = [
        "[UNK]", "this", "is", "a", "sample", "sentence", "another", 
        "example", "text", "for", "embedding", "model", "training", 
        "rust", "systems", "programming", "language", "useful", 
        "nlp", "tasks", "machine", "learning", "models", "can", 
        "process", "data"
    ]
    .iter()
    .map(|s| (s.to_string(), 1u32))
    .collect::<std::collections::HashMap<String, u32>>();

    // Create a WordPiece tokenizer with the vocabulary
    let wp_builder = WordPiece::builder()
        .unk_token("[UNK]".to_string())
        .vocab(vocab)
        .build()
        .map_err(|e| anyhow!("Failed to build WordPiece: {}", e))?;

    let mut tokenizer = Tokenizer::new(wp_builder);

    // Add a normalizer
    let normalizer = Sequence::new(vec![
        NFD.into(),
        Lowercase.into(),
        StripAccents.into(),
    ]);
    tokenizer.with_normalizer(normalizer);

    // Add a pre-tokenizer
    tokenizer.with_pre_tokenizer(Whitespace);

    Ok(tokenizer)
}

// Modify the train_or_load_model function to use FastText instead of TfidfModel
fn train_or_load_model() -> Result<FastText> {
    let model_path = Path::new("fasttext_model.bin");

    if model_path.exists() {
        // Load existing model
        println!("Loading existing FastText model");
        let mut model = FastText::new();
        // Use load_model instead of load
        model.load_model(&model_path.to_string_lossy())
            .map_err(|e| anyhow!("Failed to load FastText model: {}", e))?;
        Ok(model)
    } else {
        // Create and train a new model
        println!("Creating new FastText model");

        // Sample training data
        let training_data = vec![
            "This is a sample sentence for training",
            "Another example text for embedding model training",
            "Rust is a systems programming language",
            "Text embeddings are useful for NLP tasks",
            "Machine learning models can process text data",
        ];

        // Create a temporary training file
        let train_file = "train_data.txt";
        let mut file = File::create(train_file)?;
        for line in &training_data {
            writeln!(file, "{}", line)?;
        }

        // Train the model with skipgram
        let mut model = FastText::new();
        let mut args = fasttext::Args::new();
        // Use the correct model name from the enum (0 for skipgram)
        args.set_model(fasttext::ModelName::SG);
        args.set_dim(EMBEDDING_DIMENSION as i32); // Set embedding dimension using the constant
        args.set_epoch(5);
        args.set_input(train_file)
            .map_err(|e| anyhow!("Failed to set input file: {}", e))?;

        // Pass args as the only parameter to train
        model.train(&args)
            .map_err(|e| anyhow!("Failed to train FastText model: {}", e))?;

        // Save the model
        model.save_model(&model_path.to_string_lossy())
            .map_err(|e| anyhow!("Failed to save FastText model: {}", e))?;

        // Clean up
        std::fs::remove_file(train_file)?;

        Ok(model)
    }
}

// Structure to hold tokenized text and its embedding
struct EmbeddingResult {
    text: String,
    embedding: Vec<f32>,
}

// Create embedding for text
fn embed_text(tokenizer: &Tokenizer, model: &FastText, text: &str) -> Result<EmbeddingResult> {
    // Step 1: Tokenize the input text
    let encoding = tokenizer.encode(text, false)
        .map_err(|e| anyhow!("Failed to tokenize text: {}", e))?;

    // Step 2: Get the tokens
    let tokens: Vec<String> = encoding.get_tokens().iter()
        .map(|s| s.to_string())
        .collect();

    // Step 3: Get embeddings for these tokens
    let embedding = generate_embedding(model, &tokens);

    Ok(EmbeddingResult {
        text: text.to_string(),
        embedding,
    })
}

// Update the generate_embedding function to use FastText
fn generate_embedding(model: &FastText, tokens: &[String]) -> Vec<f32> {
    let mut embedding = vec![0.0; EMBEDDING_DIMENSION]; // Using the constant for dimension
    let mut count = 0;

    for token in tokens {
        if let Ok(token_embedding) = model.get_word_vector(token) {
            // Add token embedding to the sentence embedding
            for (i, &value) in token_embedding.iter().enumerate() {
                embedding[i] += value;
            }
            count += 1;
        }
    }

    // Average the embeddings
    if count > 0 {
        for val in &mut embedding {
            *val /= count as f32;
        }
    }

    // Normalize the embedding
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }

    embedding
}

// Function to calculate cosine similarity between two vectors
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..vec1.len().min(vec2.len()) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    // Avoid division by zero
    if norm1 > 0.0 && norm2 > 0.0 {
        dot_product / (norm1.sqrt() * norm2.sqrt())
    } else {
        0.0
    }
}

// Save embeddings to a file
fn save_embeddings_to_file(embeddings: &[EmbeddingResult], file_path: &str) -> Result<()> {
    // Convert to serializable format
    let serializable: Vec<TextEmbedding> = embeddings.iter()
        .map(|e| TextEmbedding {
            text: e.text.clone(),
            embedding: e.embedding.clone(),
        })
        .collect();

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&serializable)?;

    // Write to file
    let mut file = File::create(file_path)?;
    file.write_all(json.as_bytes())?;

    Ok(())
}
