use ndarray::{Array, ArrayD, IxDyn};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::asr::config::{ModelConfig, ModelSelection, ModelType};
use crate::asr::decoder::Vocab;
use crate::asr::AsrError;

/// Create an ONNX session with CUDA GPU acceleration
fn create_onnx_session(path: &Path, name: &str) -> Result<Session, AsrError> {
    let cuda = CUDAExecutionProvider::default()
        .with_device_id(0);
    
    log::info!("[{}] Building session with CUDA EP...", name);

    // Use with_execution_providers - the standard way to register EPs
    let session = Session::builder()
        .map_err(|e| AsrError::ModelError(format!("Session builder failed: {}", e)))?
        .with_execution_providers([
            cuda.build().error_on_failure()  // Force error if CUDA fails
        ])
        .map_err(|e| {
            log::error!("[{}] CUDA EP failed: {}. Falling back to CPU.", name, e);
            AsrError::ModelError(format!("CUDA failed: {}", e))
        })?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| AsrError::ModelError(format!("Optimization failed: {}", e)))?
        .commit_from_file(path)
        .map_err(|e| AsrError::ModelError(format!("Model load failed: {}", e)))?;

    log::info!("[{}] ✓ Session loaded with CUDA", name);
    
    Ok(session)
}

/// Transcription result
#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub tokens: Vec<u32>,
}

/// ASR Model wrapper - enum-based dispatch for different model architectures
pub enum AsrModel {
    /// Parakeet TDT (Transducer with duration)
    Parakeet(ParakeetTdtModel),
    /// Canary AED (Attention Encoder-Decoder)
    Canary(CanaryAedModel),
}

impl AsrModel {
    /// Load model from directory with selection parameters
    pub fn load(
        model_dir: &Path,
        selection: &ModelSelection,
        vocab: &Vocab,
    ) -> Result<Self, AsrError> {
        log::info!("Loading {:?} model with {:?} quantization", selection.model_type, selection.quantization);
        
        match selection.model_type {
            ModelType::Parakeet => {
                let model = ParakeetTdtModel::load(model_dir, selection)?;
                Ok(AsrModel::Parakeet(model))
            }
            ModelType::Canary => {
                let model = CanaryAedModel::load(model_dir, selection, vocab)?;
                Ok(AsrModel::Canary(model))
            }
        }
    }

    /// Run inference
    pub fn infer(
        &mut self,
        mel: &ndarray::Array2<f32>,
        source_lang: Option<&str>,
        target_lang: Option<&str>,
    ) -> Result<Vec<u32>, AsrError> {
        match self {
            AsrModel::Parakeet(model) => model.infer(mel),
            AsrModel::Canary(model) => model.infer_with_language(mel, source_lang, target_lang),
        }
    }
}

// ============================================================================
// Parakeet TDT Model
// ============================================================================

/// Parakeet TDT model - Transducer with Token and Duration prediction
pub struct ParakeetTdtModel {
    encoder: Session,
    decoder_joint: Session,
}

impl ParakeetTdtModel {
    pub fn load(model_dir: &Path, selection: &ModelSelection) -> Result<Self, AsrError> {
        let encoder_path = model_dir.join(&selection.encoder_filename());
        let decoder_path = model_dir.join(&selection.decoder_filename());

        log::info!("Loading Parakeet encoder from {:?}", encoder_path);
        let encoder = create_onnx_session(&encoder_path, "Parakeet-Encoder")?;

        log::info!("Loading Parakeet decoder from {:?}", decoder_path);
        let decoder_joint = create_onnx_session(&decoder_path, "Parakeet-Decoder")?;

        // Log model info
        log::info!("Parakeet encoder inputs: {:?}", encoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        log::info!("Parakeet decoder inputs: {:?}", decoder_joint.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());

        Ok(Self { encoder, decoder_joint })
    }

    /// Run inference on mel spectrogram
    pub fn infer(&mut self, mel: &ndarray::Array2<f32>) -> Result<Vec<u32>, AsrError> {
        // Add batch dimension: (time, mels) -> (1, mels, time)
        let mel_t = mel.t();
        let mel_3d = mel_t.insert_axis(ndarray::Axis(0));
        let mel_input: ArrayD<f32> = mel_3d.into_dyn().into_owned();

        log::info!("Parakeet encoder input shape: {:?}", mel_input.shape());

        let length = mel.shape()[0] as i64;
        let length_arr: ArrayD<i64> = ndarray::arr1(&[length]).into_dyn();

        let mel_tensor = Tensor::from_array(mel_input)
            .map_err(|e| AsrError::InferenceError(format!("Failed to create mel tensor: {}", e)))?;
        let length_tensor = Tensor::from_array(length_arr)
            .map_err(|e| AsrError::InferenceError(format!("Failed to create length tensor: {}", e)))?;

        let encoder_outputs = self.encoder
            .run(ort::inputs!["audio_signal" => mel_tensor, "length" => length_tensor])
            .map_err(|e| AsrError::InferenceError(format!("Encoder inference failed: {}", e)))?;

        let encoder_out = encoder_outputs.iter().next()
            .ok_or_else(|| AsrError::InferenceError("No encoder output".to_string()))?;

        let (enc_shape, enc_data) = encoder_out.1
            .try_extract_tensor::<f32>()
            .map_err(|e| AsrError::InferenceError(format!("Failed to extract encoder output: {}", e)))?;

        let shape: Vec<usize> = enc_shape.iter().map(|&x| x as usize).collect();
        let data: Vec<f32> = enc_data.to_vec();

        log::info!("Parakeet encoder output shape: {:?}", shape);
        drop(encoder_outputs);

        self.greedy_decode_tdt(&shape, &data)
    }

    /// Greedy decoding for TDT transducer
    fn greedy_decode_tdt(&mut self, shape: &[usize], data: &[f32]) -> Result<Vec<u32>, AsrError> {
        if shape.len() != 3 {
            return Err(AsrError::InferenceError(format!(
                "Invalid encoder output shape: {:?}", shape
            )));
        }

        let batch_size = shape[0];
        let hidden_dim = shape[1];
        let time_steps = shape[2];

        log::info!("TDT Decoding: batch={}, hidden={}, time={}", batch_size, hidden_dim, time_steps);

        // Transpose from [batch, hidden, time] to [batch, time, hidden]
        let mut encoder_transposed = vec![0.0f32; data.len()];
        for t in 0..time_steps {
            for h in 0..hidden_dim {
                encoder_transposed[t * hidden_dim + h] = data[h * time_steps + t];
            }
        }

        // Initialize LSTM states
        let state_size = 640;
        let mut states1 = vec![0.0f32; 2 * batch_size * state_size];
        let mut states2 = vec![0.0f32; 2 * batch_size * state_size];

        let vocab_size = 8193usize;
        let blank_idx = 8192i32;
        let max_tokens_per_step = 10;

        let mut decoded_tokens: Vec<i32> = Vec::new();
        let mut t = 0usize;
        let mut emitted_tokens = 0;

        while t < time_steps {
            let frame_start = t * hidden_dim;
            let encoder_frame: Vec<f32> = encoder_transposed[frame_start..frame_start + hidden_dim].to_vec();
            let target_token = decoded_tokens.last().copied().unwrap_or(blank_idx);

            let enc_frame_arr: ArrayD<f32> = Array::from_shape_vec(
                IxDyn(&[1, hidden_dim, 1]), encoder_frame
            ).map_err(|e| AsrError::InferenceError(format!("Failed to create encoder frame: {}", e)))?;

            let targets_arr: ArrayD<i32> = Array::from_shape_vec(IxDyn(&[1, 1]), vec![target_token])
                .map_err(|e| AsrError::InferenceError(format!("Failed to create targets: {}", e)))?;

            let target_length_arr: ArrayD<i32> = Array::from_shape_vec(IxDyn(&[1]), vec![1])
                .map_err(|e| AsrError::InferenceError(format!("Failed to create target_length: {}", e)))?;

            let states1_arr: ArrayD<f32> = Array::from_shape_vec(
                IxDyn(&[2, 1, state_size]), states1.clone()
            ).map_err(|e| AsrError::InferenceError(format!("Failed to create states1: {}", e)))?;

            let states2_arr: ArrayD<f32> = Array::from_shape_vec(
                IxDyn(&[2, 1, state_size]), states2.clone()
            ).map_err(|e| AsrError::InferenceError(format!("Failed to create states2: {}", e)))?;

            let enc_tensor = Tensor::from_array(enc_frame_arr)?;
            let targets_tensor = Tensor::from_array(targets_arr)?;
            let target_length_tensor = Tensor::from_array(target_length_arr)?;
            let states1_tensor = Tensor::from_array(states1_arr)?;
            let states2_tensor = Tensor::from_array(states2_arr)?;

            let decoder_outputs = self.decoder_joint.run(ort::inputs![
                "encoder_outputs" => enc_tensor,
                "targets" => targets_tensor,
                "target_length" => target_length_tensor,
                "input_states_1" => states1_tensor,
                "input_states_2" => states2_tensor
            ]).map_err(|e| AsrError::InferenceError(format!("Decoder failed at step {}: {}", t, e)))?;

            let logits_ref = decoder_outputs.get("outputs")
                .ok_or_else(|| AsrError::InferenceError("No 'outputs' in decoder result".to_string()))?;
            let (_, logits_slice) = logits_ref.try_extract_tensor::<f32>()?;
            let logits_data: Vec<f32> = logits_slice.to_vec();

            let new_states1 = if let Some(states1_ref) = decoder_outputs.get("output_states_1") {
                let (_, s) = states1_ref.try_extract_tensor::<f32>()?;
                Some(s.to_vec())
            } else { None };

            let new_states2 = if let Some(states2_ref) = decoder_outputs.get("output_states_2") {
                let (_, s) = states2_ref.try_extract_tensor::<f32>()?;
                Some(s.to_vec())
            } else { None };

            if logits_data.len() < vocab_size {
                t += 1;
                continue;
            }

            let token_logits = &logits_data[..vocab_size];
            let duration_logits = &logits_data[vocab_size..];

            let (best_token_idx, _) = token_logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            let token = best_token_idx as i32;

            let step = if !duration_logits.is_empty() {
                duration_logits.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else { 0 };

            if token != blank_idx {
                if let Some(s1) = new_states1 { states1 = s1; }
                if let Some(s2) = new_states2 { states2 = s2; }
                decoded_tokens.push(token);
                emitted_tokens += 1;
            }

            if step > 0 {
                t += step;
                emitted_tokens = 0;
            } else if token == blank_idx || emitted_tokens >= max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        log::info!("TDT decoded {} tokens", decoded_tokens.len());
        Ok(decoded_tokens.into_iter().map(|x| x as u32).collect())
    }
}

// ============================================================================
// Canary AED Model
// ============================================================================

/// Canary AED model - Attention Encoder-Decoder for multilingual ASR/translation
pub struct CanaryAedModel {
    encoder: Session,
    decoder: Session,
    /// Token -> ID mapping for special tokens
    token_to_id: HashMap<String, i64>,
    /// Special token IDs
    eos_token_id: i64,
    /// Max output sequence length
    max_sequence_length: usize,
    /// decoder_mems initial shape [num_layers, batch, 0, hidden_dim]
    decoder_mems_shape: (usize, usize),
}

impl CanaryAedModel {
    pub fn load(model_dir: &Path, selection: &ModelSelection, vocab: &Vocab) -> Result<Self, AsrError> {
        let encoder_path = model_dir.join(&selection.encoder_filename());
        let decoder_path = model_dir.join(&selection.decoder_filename());

        // Load config
        let config = ModelConfig::load(model_dir)
            .map_err(|e| AsrError::ModelError(e))?;

        log::info!("Loading Canary encoder from {:?}", encoder_path);
        let encoder = create_onnx_session(&encoder_path, "Canary-Encoder")?;

        log::info!("Loading Canary decoder from {:?}", decoder_path);
        let decoder = create_onnx_session(&decoder_path, "Canary-Decoder")?;

        // Log model info
        log::info!("Canary encoder inputs: {:?}", encoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        log::info!("Canary encoder outputs: {:?}", encoder.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());
        log::info!("Canary decoder inputs: {:?}", decoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        log::info!("Canary decoder outputs: {:?}", decoder.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());

        // Extract decoder_mems shape from model inputs
        // Shape is [num_layers, batch, seq_len, hidden_dim]
        // We need num_layers (index 0) and hidden_dim (index 3)
        let decoder_mems_shape = Self::get_decoder_mems_shape(&decoder)?;
        log::info!("Decoder mems shape: num_layers={}, hidden_dim={}", decoder_mems_shape.0, decoder_mems_shape.1);

        // Build token -> ID mapping from vocab
        let token_to_id: HashMap<String, i64> = vocab.tokens_iter()
            .map(|(&id, token)| (token.clone(), id as i64))
            .collect();

        // Get EOS token ID
        let eos_token_id = *token_to_id.get("<|endoftext|>")
            .ok_or_else(|| AsrError::VocabError("Missing <|endoftext|> token".to_string()))?;

        Ok(Self {
            encoder,
            decoder,
            token_to_id,
            eos_token_id,
            max_sequence_length: config.max_sequence_length(),
            decoder_mems_shape,
        })
    }

    /// Extract decoder_mems shape from the decoder session
    fn get_decoder_mems_shape(decoder: &Session) -> Result<(usize, usize), AsrError> {
        // Find the decoder_mems input and extract its shape
        for input in decoder.inputs.iter() {
            if input.name == "decoder_mems" {
                // input_type is ort::value::ValueType
                // Log the type for debugging
                log::info!("decoder_mems input type: {:?}", input.input_type);
                
                // Use the tensor_shape() method to get the Shape
                if let Some(shape) = input.input_type.tensor_shape() {
                    // Shape dereferences to &[i64], where -1 means dynamic
                    // Shape is [num_layers, batch, seq_len, hidden_dim]
                    let dims: &[i64] = &**shape;
                    if dims.len() >= 4 {
                        // num_layers and hidden_dim should be fixed (positive)
                        let num_layers = if dims[0] > 0 { dims[0] as usize } else { 10 };
                        let hidden_dim = if dims[3] > 0 { dims[3] as usize } else { 1024 };
                        log::info!("Extracted decoder_mems shape: num_layers={}, hidden_dim={}", num_layers, hidden_dim);
                        return Ok((num_layers, hidden_dim));
                    }
                }
            }
        }
        // Fallback to reasonable defaults if we can't find the shape
        log::warn!("Could not determine decoder_mems shape, using defaults");
        Ok((10, 1024))
    }

    fn build_transcribe_prefix(
        token_to_id: &HashMap<String, i64>,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<Vec<i64>, AsrError> {
        let get_token = |name: &str| -> Result<i64, AsrError> {
            token_to_id.get(name)
                .copied()
                .ok_or_else(|| AsrError::VocabError(format!("Missing token: {}", name)))
        };

        // Canary prefix format:
        // [startofcontext, startoftranscript, emo:undefined, source_lang, target_lang, pnc, noitn, notimestamp, nodiarize]
        Ok(vec![
            get_token("<|startofcontext|>")?,
            get_token("<|startoftranscript|>")?,
            get_token("<|emo:undefined|>")?,
            get_token(&format!("<|{}|>", source_lang))?,
            get_token(&format!("<|{}|>", target_lang))?,
            get_token("<|pnc|>")?,
            get_token("<|noitn|>")?,
            get_token("<|notimestamp|>")?,
            get_token("<|nodiarize|>")?,
        ])
    }

    pub fn infer_with_language(
        &mut self,
        mel: &ndarray::Array2<f32>,
        source_lang: Option<&str>,
        target_lang: Option<&str>,
    ) -> Result<Vec<u32>, AsrError> {
        let source = source_lang.unwrap_or("en");
        let target = target_lang.unwrap_or(source);

        log::info!("Canary inference: source={}, target={}", source, target);

        // Build prefix for this language pair
        let prefix = Self::build_transcribe_prefix(&self.token_to_id, source, target)?;

        // Add batch dimension: (time, mels) -> (1, mels, time)
        let mel_t = mel.t();
        let mel_3d = mel_t.insert_axis(ndarray::Axis(0));
        let mel_input: ArrayD<f32> = mel_3d.into_dyn().into_owned();

        log::info!("Canary encoder input shape: {:?}", mel_input.shape());

        let length = mel.shape()[0] as i64;
        let length_arr: ArrayD<i64> = ndarray::arr1(&[length]).into_dyn();

        let mel_tensor = Tensor::from_array(mel_input)
            .map_err(|e| AsrError::InferenceError(format!("Failed to create mel tensor: {}", e)))?;
        let length_tensor = Tensor::from_array(length_arr)
            .map_err(|e| AsrError::InferenceError(format!("Failed to create length tensor: {}", e)))?;

        // Run encoder
        let encoder_outputs = self.encoder
            .run(ort::inputs!["audio_signal" => mel_tensor, "length" => length_tensor])
            .map_err(|e| AsrError::InferenceError(format!("Encoder failed: {}", e)))?;

        // Extract encoder embeddings and mask
        let embeddings_ref = encoder_outputs.get("encoder_embeddings")
            .ok_or_else(|| AsrError::InferenceError("No encoder_embeddings output".to_string()))?;
        let mask_ref = encoder_outputs.get("encoder_mask")
            .ok_or_else(|| AsrError::InferenceError("No encoder_mask output".to_string()))?;

        let (emb_shape, emb_data) = embeddings_ref.try_extract_tensor::<f32>()?;
        let (mask_shape, mask_data) = mask_ref.try_extract_tensor::<i64>()?;

        let embeddings: Vec<f32> = emb_data.to_vec();
        let encoder_mask: Vec<i64> = mask_data.to_vec();

        let emb_dims: Vec<usize> = emb_shape.iter().map(|&x| x as usize).collect();
        let mask_dims: Vec<usize> = mask_shape.iter().map(|&x| x as usize).collect();

        log::info!("Encoder embeddings shape: {:?}", emb_dims);
        log::info!("Encoder mask shape: {:?}", mask_dims);

        drop(encoder_outputs);

        // Run autoregressive decoding
        self.autoregressive_decode(prefix, embeddings, emb_dims, encoder_mask, mask_dims)
    }

    fn autoregressive_decode(
        &mut self,
        prefix: Vec<i64>,
        encoder_embeddings: Vec<f32>,
        emb_dims: Vec<usize>,
        encoder_mask: Vec<i64>,
        mask_dims: Vec<usize>,
    ) -> Result<Vec<u32>, AsrError> {
        let batch_size = emb_dims[0];
        let prefix_len = prefix.len();
        let mut tokens = prefix;

        let (num_layers, hidden_dim) = self.decoder_mems_shape;

        // Create encoder embeddings array - reused every iteration
        let emb_arr: ArrayD<f32> = Array::from_shape_vec(IxDyn(&emb_dims), encoder_embeddings)
            .map_err(|e| AsrError::InferenceError(format!("Failed to create embeddings: {}", e)))?;
        let mask_arr: ArrayD<i64> = Array::from_shape_vec(IxDyn(&mask_dims), encoder_mask)
            .map_err(|e| AsrError::InferenceError(format!("Failed to create mask: {}", e)))?;

        // Initialize empty KV cache: [num_layers, batch, 0, hidden_dim]
        let mut decoder_mems: ArrayD<f32> = Array::from_shape_vec(
            IxDyn(&[num_layers, batch_size, 0, hidden_dim]),
            vec![]
        ).unwrap();

        let decode_start = Instant::now();
        let mut step = 0usize;

        // Warm-up run to ensure GPU kernels are compiled
        log::info!("Starting autoregressive decode (prefix_len={}, max_seq={})", prefix_len, self.max_sequence_length);

        while tokens.len() < self.max_sequence_length {
            let step_start = Instant::now();

            // KEY OPTIMIZATION: Pass all tokens on first step, only LAST token after
            // This is how KV cache works - we don't reprocess old tokens
            let decoder_mems_seq_len = decoder_mems.shape()[2];
            let input_ids: Vec<i64> = if decoder_mems_seq_len == 0 {
                tokens.clone()  // First step: all prefix tokens
            } else {
                vec![*tokens.last().unwrap()]  // Subsequent: only new token
            };

            let input_ids_arr: ArrayD<i64> = Array::from_shape_vec(
                IxDyn(&[1, input_ids.len()]),
                input_ids.clone()
            ).map_err(|e| AsrError::InferenceError(format!("Failed to create input_ids: {}", e)))?;

            // Create tensors for this iteration
            let input_tensor = Tensor::from_array(input_ids_arr)?;
            let emb_tensor = Tensor::from_array(emb_arr.clone())?;
            let mask_tensor = Tensor::from_array(mask_arr.clone())?;
            let mems_tensor = Tensor::from_array(decoder_mems.clone())?;

            // Run decoder - simple session.run() like Python reference
            let outputs = self.decoder.run(ort::inputs![
                "input_ids" => input_tensor,
                "encoder_embeddings" => emb_tensor,
                "encoder_mask" => mask_tensor,
                "decoder_mems" => mems_tensor
            ]).map_err(|e| AsrError::InferenceError(format!("Decoder failed at step {}: {}", step, e)))?;

            // Extract logits
            let logits_ref = outputs.get("logits")
                .ok_or_else(|| AsrError::InferenceError("No logits output".to_string()))?;
            let (logits_shape, logits_data) = logits_ref.try_extract_tensor::<f32>()?;

            // Argmax on last position
            let vocab_size = logits_shape[2] as usize;
            let seq_len = logits_shape[1] as usize;
            let last_offset = (seq_len - 1) * vocab_size;

            let next_token = logits_data.iter()
                .skip(last_offset)
                .take(vocab_size)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(0);

            // Log first few steps with timing
            if step < 5 || step % 50 == 0 {
                log::info!(
                    "Step {}: input_len={}, mems_seq={}, time={:.1}ms",
                    step, input_ids.len(), decoder_mems_seq_len, step_start.elapsed().as_secs_f32() * 1000.0
                );
            }

            // Check for EOS
            if next_token == self.eos_token_id {
                let total = decode_start.elapsed().as_secs_f32();
                log::info!("EOS at step {} after {:.2}s ({:.1}ms/token)", step, total, total * 1000.0 / step as f32);
                break;
            }

            tokens.push(next_token);

            // Extract new KV cache (decoder_hidden_states)
            let hs_ref = outputs.get("decoder_hidden_states")
                .ok_or_else(|| AsrError::InferenceError("No decoder_hidden_states output".to_string()))?;
            let (hs_shape, hs_data) = hs_ref.try_extract_tensor::<f32>()?;

            // Update decoder_mems for next iteration
            let hs_dims: Vec<usize> = hs_shape.iter().map(|&x| x as usize).collect();
            decoder_mems = Array::from_shape_vec(IxDyn(&hs_dims), hs_data.to_vec())
                .map_err(|e| AsrError::InferenceError(format!("Failed to update mems: {}", e)))?;

            step += 1;
        }

        let total_time = decode_start.elapsed().as_secs_f32();
        let tokens_generated = tokens.len() - prefix_len;
        log::info!(
            "Decoded {} tokens in {:.2}s ({:.1}ms/token avg)",
            tokens_generated,
            total_time,
            if tokens_generated > 0 { total_time * 1000.0 / tokens_generated as f32 } else { 0.0 }
        );

        // Return tokens (skip prefix, filter EOS)
        Ok(tokens[prefix_len..]
            .iter()
            .filter(|&&t| t != self.eos_token_id)
            .map(|&t| t as u32)
            .collect())
    }
}

impl From<ort::Error> for AsrError {
    fn from(e: ort::Error) -> Self {
        AsrError::InferenceError(e.to_string())
    }
}
