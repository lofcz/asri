use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::asr::AsrError;

/// Vocabulary for token-to-text conversion
pub struct Vocab {
    tokens: HashMap<u32, String>,
    special_tokens: Vec<u32>,
}

impl Vocab {
    /// Load vocabulary from file
    pub fn load(path: &Path) -> Result<Self, AsrError> {
        let file = File::open(path).map_err(|e| {
            AsrError::VocabError(format!("Failed to open vocab file {:?}: {}", path, e))
        })?;

        let reader = BufReader::new(file);
        let mut tokens = HashMap::new();
        let mut special_tokens = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| AsrError::VocabError(format!("Failed to read line: {}", e)))?;
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            
            if parts.len() >= 2 {
                let token = parts[0].to_string();
                let id: u32 = parts[1].parse().map_err(|e| {
                    AsrError::VocabError(format!("Failed to parse token id: {}", e))
                })?;

                // Track special tokens (those in angle brackets, including <|...|> format)
                if (token.starts_with('<') && token.ends_with('>')) ||
                   (token.starts_with("<|") && token.ends_with("|>")) {
                    special_tokens.push(id);
                }

                tokens.insert(id, token);
            }
        }

        Ok(Self {
            tokens,
            special_tokens,
        })
    }

    /// Get token string for an ID
    pub fn get(&self, id: u32) -> Option<&str> {
        self.tokens.get(&id).map(|s| s.as_str())
    }

    /// Check if token is special
    pub fn is_special(&self, id: u32) -> bool {
        self.special_tokens.contains(&id)
    }

    /// Get vocabulary size
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Get iterator over all tokens
    pub fn tokens_iter(&self) -> impl Iterator<Item = (&u32, &String)> {
        self.tokens.iter()
    }
}

/// Greedy decoder for ASR output
pub struct GreedyDecoder<'a> {
    vocab: &'a Vocab,
    blank_idx: u32,
}

impl<'a> GreedyDecoder<'a> {
    pub fn new(vocab: &'a Vocab) -> Self {
        // Blank token is at the end of vocab for Parakeet (8192)
        // For Canary (AED), there's no blank token, but this won't matter
        // since we filter by ID matching
        let blank_idx = vocab.len() as u32 - 1;
        Self { vocab, blank_idx }
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        // Collect all token strings
        // SentencePiece uses ▁ (U+2581) to indicate word boundaries (space before token)
        let token_strings: Vec<String> = tokens
            .iter()
            .filter(|&&id| id != self.blank_idx) // Skip blank (for transducer models)
            .filter(|&&id| !self.vocab.is_special(id)) // Skip special tokens
            .filter_map(|&id| self.vocab.get(id))
            .map(|s| s.replace('▁', " ")) // Replace SentencePiece space marker
            .collect();

        // Join all tokens (no separator - spaces are encoded in the tokens themselves)
        let joined = token_strings.join("");

        // Simple cleanup: trim and normalize multiple spaces
        self.normalize_text(&joined)
    }

    /// Normalize text - just clean up whitespace without breaking punctuation spacing
    fn normalize_text(&self, text: &str) -> String {
        // Split on whitespace and rejoin with single spaces
        // This handles multiple consecutive spaces and trims
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}
