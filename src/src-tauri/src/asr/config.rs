use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Model types supported by the ASR engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// NeMo Parakeet TDT (Transducer) - English only, fast
    Parakeet,
    /// NeMo Canary AED (Attention Encoder-Decoder) - Multilingual with translation
    Canary,
}

impl Default for ModelType {
    fn default() -> Self {
        Self::Parakeet
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Parakeet => write!(f, "parakeet"),
            ModelType::Canary => write!(f, "canary"),
        }
    }
}

/// Quantization options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// Full precision (fp32)
    #[default]
    Full,
    /// INT8 quantization (smaller, faster)
    Int8,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Quantization::Full => write!(f, "full"),
            Quantization::Int8 => write!(f, "int8"),
        }
    }
}

/// Language code for multilingual models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Language(pub String);

impl Default for Language {
    fn default() -> Self {
        Self("en".to_string())
    }
}

impl Language {
    pub fn new(code: &str) -> Self {
        Self(code.to_lowercase())
    }
    
    pub fn code(&self) -> &str {
        &self.0
    }
}

/// Model configuration loaded from config.json
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[allow(dead_code)]
    model_type: Option<String>,
    #[allow(dead_code)]
    features_size: Option<usize>,
    max_sequence_length: Option<usize>,
    #[allow(dead_code)]
    subsampling_factor: Option<usize>,
}

impl ModelConfig {
    /// Load configuration from model directory
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            let config_str = fs::read_to_string(&config_path)
                .map_err(|e| format!("Failed to read config.json: {}", e))?;
            serde_json::from_str(&config_str)
                .map_err(|e| format!("Failed to parse config.json: {}", e))
        } else {
            // Default config
            Ok(Self {
                model_type: None,
                features_size: Some(80),
                max_sequence_length: Some(1024),
                subsampling_factor: Some(8),
            })
        }
    }

    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length.unwrap_or(1024)
    }
}

/// Model selection configuration for initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelection {
    pub model_type: ModelType,
    pub quantization: Quantization,
    pub source_language: Language,
    pub target_language: Option<Language>,
}

impl Default for ModelSelection {
    fn default() -> Self {
        Self {
            model_type: ModelType::Parakeet,
            quantization: Quantization::Full,
            source_language: Language::default(),
            target_language: None,
        }
    }
}

impl ModelSelection {
    /// Get the encoder model filename based on quantization
    pub fn encoder_filename(&self) -> String {
        match self.quantization {
            Quantization::Full => "encoder-model.onnx".to_string(),
            Quantization::Int8 => "encoder-model.int8.onnx".to_string(),
        }
    }

    /// Get the decoder model filename based on model type and quantization
    pub fn decoder_filename(&self) -> String {
        match (self.model_type, self.quantization) {
            (ModelType::Parakeet, Quantization::Full) => "decoder_joint-model.onnx".to_string(),
            (ModelType::Parakeet, Quantization::Int8) => "decoder_joint-model.int8.onnx".to_string(),
            (ModelType::Canary, Quantization::Full) => "decoder-model.onnx".to_string(),
            (ModelType::Canary, Quantization::Int8) => "decoder-model.int8.onnx".to_string(),
        }
    }

}

/// Available model information for frontend
#[derive(Debug, Clone, Serialize)]
pub struct AvailableModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub model_type: ModelType,
    pub supports_languages: bool,
    pub supported_languages: Vec<LanguageInfo>,
    pub quantizations: Vec<QuantizationInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LanguageInfo {
    pub code: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizationInfo {
    pub id: String,
    pub name: String,
    pub description: String,
}

/// Get list of available models
pub fn get_available_models() -> Vec<AvailableModel> {
    vec![
        AvailableModel {
            id: "parakeet".to_string(),
            name: "Parakeet TDT 0.6B".to_string(),
            description: "Fast English-only speech recognition".to_string(),
            model_type: ModelType::Parakeet,
            supports_languages: false,
            supported_languages: vec![
                LanguageInfo { code: "en".to_string(), name: "English".to_string() },
            ],
            quantizations: vec![
                QuantizationInfo {
                    id: "full".to_string(),
                    name: "Full Precision".to_string(),
                    description: "Best quality, larger model".to_string(),
                },
                QuantizationInfo {
                    id: "int8".to_string(),
                    name: "INT8".to_string(),
                    description: "Faster, smaller model".to_string(),
                },
            ],
        },
        AvailableModel {
            id: "canary".to_string(),
            name: "Canary 1B v2".to_string(),
            description: "Multilingual ASR & translation for 25 languages".to_string(),
            model_type: ModelType::Canary,
            supports_languages: true,
            supported_languages: get_canary_languages(),
            quantizations: vec![
                QuantizationInfo {
                    id: "full".to_string(),
                    name: "Full Precision".to_string(),
                    description: "Best quality, larger model".to_string(),
                },
                QuantizationInfo {
                    id: "int8".to_string(),
                    name: "INT8".to_string(),
                    description: "Faster, smaller model".to_string(),
                },
            ],
        },
    ]
}

/// Get Canary supported languages
fn get_canary_languages() -> Vec<LanguageInfo> {
    vec![
        LanguageInfo { code: "en".to_string(), name: "English".to_string() },
        LanguageInfo { code: "de".to_string(), name: "German".to_string() },
        LanguageInfo { code: "fr".to_string(), name: "French".to_string() },
        LanguageInfo { code: "es".to_string(), name: "Spanish".to_string() },
        LanguageInfo { code: "it".to_string(), name: "Italian".to_string() },
        LanguageInfo { code: "pt".to_string(), name: "Portuguese".to_string() },
        LanguageInfo { code: "nl".to_string(), name: "Dutch".to_string() },
        LanguageInfo { code: "pl".to_string(), name: "Polish".to_string() },
        LanguageInfo { code: "ru".to_string(), name: "Russian".to_string() },
        LanguageInfo { code: "uk".to_string(), name: "Ukrainian".to_string() },
        LanguageInfo { code: "bg".to_string(), name: "Bulgarian".to_string() },
        LanguageInfo { code: "cs".to_string(), name: "Czech".to_string() },
        LanguageInfo { code: "da".to_string(), name: "Danish".to_string() },
        LanguageInfo { code: "el".to_string(), name: "Greek".to_string() },
        LanguageInfo { code: "et".to_string(), name: "Estonian".to_string() },
        LanguageInfo { code: "fi".to_string(), name: "Finnish".to_string() },
        LanguageInfo { code: "hr".to_string(), name: "Croatian".to_string() },
        LanguageInfo { code: "hu".to_string(), name: "Hungarian".to_string() },
        LanguageInfo { code: "lt".to_string(), name: "Lithuanian".to_string() },
        LanguageInfo { code: "lv".to_string(), name: "Latvian".to_string() },
        LanguageInfo { code: "mt".to_string(), name: "Maltese".to_string() },
        LanguageInfo { code: "ro".to_string(), name: "Romanian".to_string() },
        LanguageInfo { code: "sk".to_string(), name: "Slovak".to_string() },
        LanguageInfo { code: "sl".to_string(), name: "Slovenian".to_string() },
        LanguageInfo { code: "sv".to_string(), name: "Swedish".to_string() },
    ]
}
