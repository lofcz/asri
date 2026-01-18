pub mod audio;
pub mod config;
pub mod decoder;
pub mod model;

use std::path::PathBuf;

use crate::asr::config::{ModelSelection, ModelType};
use crate::asr::decoder::Vocab;
use crate::asr::model::AsrModel;

pub use audio::AudioProcessor;
pub use config::{get_available_models, AvailableModel};
pub use decoder::GreedyDecoder;
pub use model::TranscriptionResult;

/// Global ASR engine state
pub struct AsrEngine {
    model: Option<AsrModel>,
    vocab: Option<Vocab>,
    audio_processor: AudioProcessor,
    current_selection: Option<ModelSelection>,
    models_root: PathBuf,
}

impl AsrEngine {
    pub fn new(models_root: PathBuf) -> Self {
        Self {
            model: None,
            vocab: None,
            audio_processor: AudioProcessor::new(),
            current_selection: None,
            models_root,
        }
    }

    /// Get the model directory for a specific model type
    fn model_dir(&self, model_type: ModelType) -> PathBuf {
        self.models_root.join(model_type.to_string())
    }

    /// Initialize the ASR engine with the specified model selection
    pub fn initialize(&mut self, selection: ModelSelection) -> Result<(), AsrError> {
        let model_dir = self.model_dir(selection.model_type);
        log::info!("Initializing ASR engine from {:?} with {:?}", model_dir, selection);

        // Load vocabulary first
        let vocab_path = model_dir.join("vocab.txt");
        let vocab = Vocab::load(&vocab_path)?;
        log::info!("Loaded vocabulary with {} tokens", vocab.len());

        // Load ONNX model with the vocab for token mapping (needed for Canary)
        let model = AsrModel::load(&model_dir, &selection, &vocab)?;
        log::info!("ASR model loaded successfully: {:?}", selection.model_type);

        self.vocab = Some(vocab);
        self.model = Some(model);
        self.current_selection = Some(selection);

        Ok(())
    }

    /// Check if the engine is initialized
    pub fn is_initialized(&self) -> bool {
        self.model.is_some() && self.vocab.is_some()
    }

    /// Get current model selection
    pub fn current_selection(&self) -> Option<&ModelSelection> {
        self.current_selection.as_ref()
    }

    /// Transcribe with language options
    pub fn transcribe_with_language(
        &mut self,
        audio_bytes: &[u8],
        source_lang: Option<&str>,
        target_lang: Option<&str>,
    ) -> Result<TranscriptionResult, AsrError> {
        if !self.is_initialized() {
            return Err(AsrError::NotInitialized);
        }

        // Process audio to get mel spectrogram
        let mel = self.audio_processor.process_audio(audio_bytes)?;
        log::info!("Generated mel spectrogram: {:?}", mel.shape());

        // Run inference with language options
        let model = self.model.as_mut().ok_or(AsrError::NotInitialized)?;
        let tokens = model.infer(&mel, source_lang, target_lang)?;
        log::info!("Got {} tokens from inference", tokens.len());

        // Decode tokens to text
        let vocab = self.vocab.as_ref().ok_or(AsrError::NotInitialized)?;
        let decoder = GreedyDecoder::new(vocab);
        let text = decoder.decode(&tokens);

        Ok(TranscriptionResult { text, tokens })
    }

    /// Transcribe raw audio samples with language options
    pub fn transcribe_raw_with_language(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        source_lang: Option<&str>,
        target_lang: Option<&str>,
    ) -> Result<TranscriptionResult, AsrError> {
        if !self.is_initialized() {
            return Err(AsrError::NotInitialized);
        }

        // Process raw samples to get mel spectrogram
        let mel = self.audio_processor.process_raw_samples(samples, sample_rate)?;
        log::info!("Generated mel spectrogram from raw samples: {:?}", mel.shape());

        // Run inference with language options
        let model = self.model.as_mut().ok_or(AsrError::NotInitialized)?;
        let tokens = model.infer(&mel, source_lang, target_lang)?;
        log::info!("Got {} tokens from inference", tokens.len());

        // Decode tokens to text
        let vocab = self.vocab.as_ref().ok_or(AsrError::NotInitialized)?;
        let decoder = GreedyDecoder::new(vocab);
        let text = decoder.decode(&tokens);

        Ok(TranscriptionResult { text, tokens })
    }
}

/// ASR-related errors
#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("ASR engine not initialized")]
    NotInitialized,

    #[error("Failed to load vocabulary: {0}")]
    VocabError(String),

    #[error("Failed to load model: {0}")]
    ModelError(String),

    #[error("Audio processing error: {0}")]
    AudioError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl serde::Serialize for AsrError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
