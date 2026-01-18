mod asr;

use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;
use tokio::sync::RwLock;

use asr::{AsrEngine, AsrError, TranscriptionResult, get_available_models, AvailableModel};
use asr::config::{ModelSelection, ModelType, Quantization, Language};

/// Shared ASR engine state
pub struct AppState {
    asr_engine: Arc<RwLock<AsrEngine>>,
}

/// Model initialization parameters from frontend
#[derive(serde::Deserialize)]
pub struct InitModelParams {
    model_type: String,
    quantization: String,
    source_language: Option<String>,
    target_language: Option<String>,
}

/// Transcription parameters from frontend
#[derive(serde::Deserialize)]
pub struct TranscribeParams {
    source_language: Option<String>,
    target_language: Option<String>,
}

/// Current model status for frontend
#[derive(serde::Serialize)]
pub struct ModelStatus {
    is_loaded: bool,
    model_type: Option<String>,
    quantization: Option<String>,
    source_language: Option<String>,
    target_language: Option<String>,
}

/// Get list of available models
#[tauri::command]
fn get_models() -> Vec<AvailableModel> {
    get_available_models()
}

/// Initialize the ASR model with specific configuration
#[tauri::command]
async fn initialize_model(
    params: InitModelParams,
    state: State<'_, AppState>,
) -> Result<String, AsrError> {
    let mut engine = state.asr_engine.write().await;

    // Parse model type
    let model_type = match params.model_type.to_lowercase().as_str() {
        "parakeet" => ModelType::Parakeet,
        "canary" => ModelType::Canary,
        _ => return Err(AsrError::ModelError(format!("Unknown model type: {}", params.model_type))),
    };

    // Parse quantization
    let quantization = match params.quantization.to_lowercase().as_str() {
        "full" => Quantization::Full,
        "int8" => Quantization::Int8,
        _ => return Err(AsrError::ModelError(format!("Unknown quantization: {}", params.quantization))),
    };

    let selection = ModelSelection {
        model_type,
        quantization,
        source_language: Language::new(params.source_language.as_deref().unwrap_or("en")),
        target_language: params.target_language.map(|l| Language::new(&l)),
    };

    engine.initialize(selection)?;
    Ok(format!("Model {} ({}) initialized successfully", params.model_type, params.quantization))
}

/// Check if the model is initialized and get status
#[tauri::command]
async fn get_model_status(state: State<'_, AppState>) -> Result<ModelStatus, AsrError> {
    let engine = state.asr_engine.read().await;
    
    if let Some(selection) = engine.current_selection() {
        Ok(ModelStatus {
            is_loaded: engine.is_initialized(),
            model_type: Some(selection.model_type.to_string()),
            quantization: Some(selection.quantization.to_string()),
            source_language: Some(selection.source_language.code().to_string()),
            target_language: selection.target_language.as_ref().map(|l| l.code().to_string()),
        })
    } else {
        Ok(ModelStatus {
            is_loaded: false,
            model_type: None,
            quantization: None,
            source_language: None,
            target_language: None,
        })
    }
}

/// Legacy: Check if the model is initialized (for backwards compatibility)
#[tauri::command]
async fn is_model_ready(state: State<'_, AppState>) -> Result<bool, AsrError> {
    let engine = state.asr_engine.read().await;
    Ok(engine.is_initialized())
}

/// Transcribe audio bytes (from file)
#[tauri::command]
async fn transcribe_audio(
    audio_bytes: Vec<u8>,
    params: Option<TranscribeParams>,
    state: State<'_, AppState>,
) -> Result<TranscriptionResult, AsrError> {
    let mut engine = state.asr_engine.write().await;

    if !engine.is_initialized() {
        return Err(AsrError::NotInitialized);
    }

    log::info!("Transcribing {} bytes of audio", audio_bytes.len());

    let (source_lang, target_lang) = params
        .map(|p| (p.source_language, p.target_language))
        .unwrap_or((None, None));

    let result = engine.transcribe_with_language(
        &audio_bytes,
        source_lang.as_deref(),
        target_lang.as_deref(),
    )?;

    log::info!("Transcription complete: {}", result.text);
    Ok(result)
}

/// Transcribe raw audio samples (from microphone via Web Audio API)
#[tauri::command]
async fn transcribe_raw_audio(
    samples: Vec<f32>,
    sample_rate: u32,
    params: Option<TranscribeParams>,
    state: State<'_, AppState>,
) -> Result<TranscriptionResult, AsrError> {
    let mut engine = state.asr_engine.write().await;

    if !engine.is_initialized() {
        return Err(AsrError::NotInitialized);
    }

    log::info!("Transcribing {} raw samples at {} Hz", samples.len(), sample_rate);

    let (source_lang, target_lang) = params
        .map(|p| (p.source_language, p.target_language))
        .unwrap_or((None, None));

    let result = engine.transcribe_raw_with_language(
        &samples,
        sample_rate,
        source_lang.as_deref(),
        target_lang.as_deref(),
    )?;

    log::info!("Transcription complete: {}", result.text);
    Ok(result)
}

/// Get supported audio formats
#[tauri::command]
fn get_supported_formats() -> Vec<String> {
    vec![
        "wav".to_string(),
        "mp3".to_string(),
        "flac".to_string(),
        "ogg".to_string(),
        "m4a".to_string(),
        "aac".to_string(),
    ]
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Determine models root directory
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Try to find models directory by going up the directory tree
    let models_root = cwd
        .ancestors()
        .find_map(|p| {
            let candidate = p.join("models");
            if candidate.exists() {
                Some(candidate)
            } else {
                None
            }
        })
        .unwrap_or_else(|| cwd.join("models"));

    log::info!("Models root directory: {:?}", models_root);
    log::info!("Models directory exists: {}", models_root.exists());

    // List available models
    if models_root.exists() {
        if let Ok(entries) = std::fs::read_dir(&models_root) {
            for entry in entries.flatten() {
                log::info!("  Found model directory: {:?}", entry.file_name());
            }
        }
    }

    let app_state = AppState {
        asr_engine: Arc::new(RwLock::new(AsrEngine::new(models_root))),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            get_models,
            initialize_model,
            get_model_status,
            is_model_ready,
            transcribe_audio,
            transcribe_raw_audio,
            get_supported_formats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
