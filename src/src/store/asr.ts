import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { invoke } from "@tauri-apps/api/core";

export interface TranscriptionResult {
  text: string;
  tokens: number[];
}

export interface LanguageInfo {
  code: string;
  name: string;
}

export interface QuantizationInfo {
  id: string;
  name: string;
  description: string;
}

export interface AvailableModel {
  id: string;
  name: string;
  description: string;
  model_type: string;
  supports_languages: boolean;
  supported_languages: LanguageInfo[];
  quantizations: QuantizationInfo[];
}

export interface ModelStatus {
  is_loaded: boolean;
  model_type: string | null;
  quantization: string | null;
  source_language: string | null;
  target_language: string | null;
}

export interface ASRState {
  // State
  isModelLoaded: boolean;
  isModelLoading: boolean;
  isRecording: boolean;
  isProcessing: boolean;
  transcript: string;
  error: string | null;
  audioBlob: Blob | null;
  recordingDuration: number;

  // Model selection state
  availableModels: AvailableModel[];
  selectedModelId: string;
  selectedQuantization: string;
  sourceLanguage: string;
  targetLanguage: string | null;
  modelStatus: ModelStatus | null;

  // Actions
  fetchAvailableModels: () => Promise<void>;
  initializeModel: () => Promise<void>;
  setSelectedModel: (modelId: string) => void;
  setSelectedQuantization: (quantization: string) => void;
  setSourceLanguage: (language: string) => void;
  setTargetLanguage: (language: string | null) => void;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  transcribeAudio: (audioData: ArrayBuffer) => Promise<void>;
  transcribeRawAudio: (samples: Float32Array, sampleRate: number) => Promise<void>;
  transcribeFile: (file: File) => Promise<void>;
  clearTranscript: () => void;
  clearError: () => void;
  setRecordingDuration: (duration: number) => void;
}

// MediaRecorder instance stored outside of state
let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let recordingStartTime: number = 0;
let durationInterval: ReturnType<typeof setInterval> | null = null;

// Audio context for decoding
let audioContext: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (!audioContext) {
    audioContext = new AudioContext({ sampleRate: 16000 });
  }
  return audioContext;
}

// Convert audio blob to raw PCM samples at 16kHz mono
async function decodeAudioBlob(blob: Blob): Promise<{ samples: Float32Array; sampleRate: number }> {
  const ctx = getAudioContext();
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  
  // Get samples from first channel (mono)
  const samples = audioBuffer.getChannelData(0);
  
  return {
    samples: new Float32Array(samples),
    sampleRate: audioBuffer.sampleRate,
  };
}

export const useASRStore = create<ASRState>()(
  immer((set, get) => ({
    // Initial state
    isModelLoaded: false,
    isModelLoading: false,
    isRecording: false,
    isProcessing: false,
    transcript: "",
    error: null,
    audioBlob: null,
    recordingDuration: 0,

    // Model selection state
    availableModels: [],
    selectedModelId: "parakeet",
    selectedQuantization: "full",
    sourceLanguage: "en",
    targetLanguage: null,
    modelStatus: null,

    // Fetch available models from backend
    fetchAvailableModels: async () => {
      try {
        const models = await invoke<AvailableModel[]>("get_models");
        set((state) => {
          state.availableModels = models;
        });
      } catch (err) {
        console.error("Failed to fetch models:", err);
      }
    },

    // Initialize the ASR model with current selection
    initializeModel: async () => {
      set((state) => {
        state.isModelLoading = true;
        state.error = null;
      });

      try {
        const { selectedModelId, selectedQuantization, sourceLanguage, targetLanguage } = get();

        await invoke<string>("initialize_model", {
          params: {
            model_type: selectedModelId,
            quantization: selectedQuantization,
            source_language: sourceLanguage,
            target_language: targetLanguage,
          },
        });

        // Fetch updated status
        const status = await invoke<ModelStatus>("get_model_status");

        set((state) => {
          state.isModelLoaded = true;
          state.isModelLoading = false;
          state.modelStatus = status;
        });
      } catch (err) {
        set((state) => {
          state.error = `Failed to initialize model: ${err}`;
          state.isModelLoading = false;
        });
      }
    },

    // Model selection setters
    setSelectedModel: (modelId: string) => {
      set((state) => {
        state.selectedModelId = modelId;
        // Reset to default language when switching models
        state.sourceLanguage = "en";
        state.targetLanguage = null;
        // Mark model as not loaded since we changed selection
        state.isModelLoaded = false;
        state.modelStatus = null;
      });
    },

    setSelectedQuantization: (quantization: string) => {
      set((state) => {
        state.selectedQuantization = quantization;
        // Mark model as not loaded since we changed selection
        state.isModelLoaded = false;
        state.modelStatus = null;
      });
    },

    setSourceLanguage: (language: string) => {
      set((state) => {
        state.sourceLanguage = language;
      });
    },

    setTargetLanguage: (language: string | null) => {
      set((state) => {
        state.targetLanguage = language;
      });
    },

    // Start microphone recording
    startRecording: async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });

        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        });

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          stream.getTracks().forEach((track) => track.stop());

          if (durationInterval) {
            clearInterval(durationInterval);
            durationInterval = null;
          }

          const audioBlob = new Blob(audioChunks, { type: "audio/webm" });

          set((state) => {
            state.audioBlob = audioBlob;
            state.isRecording = false;
          });

          try {
            const { samples, sampleRate } = await decodeAudioBlob(audioBlob);
            get().transcribeRawAudio(samples, sampleRate);
          } catch (err) {
            set((state) => {
              state.error = `Failed to decode audio: ${err}`;
              state.isProcessing = false;
            });
          }
        };

        mediaRecorder.start(100);
        recordingStartTime = Date.now();

        durationInterval = setInterval(() => {
          const duration = (Date.now() - recordingStartTime) / 1000;
          set((state) => {
            state.recordingDuration = duration;
          });
        }, 100);

        set((state) => {
          state.isRecording = true;
          state.error = null;
          state.recordingDuration = 0;
        });
      } catch (err) {
        set((state) => {
          state.error = `Failed to start recording: ${err}`;
        });
      }
    },

    // Stop recording
    stopRecording: async () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
    },

    // Transcribe raw audio samples (from microphone via Web Audio API)
    transcribeRawAudio: async (samples: Float32Array, sampleRate: number) => {
      set((state) => {
        state.isProcessing = true;
        state.error = null;
      });

      try {
        const { sourceLanguage, targetLanguage } = get();
        const samplesArray = Array.from(samples);

        const result = await invoke<TranscriptionResult>("transcribe_raw_audio", {
          samples: samplesArray,
          sampleRate,
          params: {
            source_language: sourceLanguage,
            target_language: targetLanguage,
          },
        });

        set((state) => {
          state.transcript = result.text;
          state.isProcessing = false;
        });
      } catch (err) {
        set((state) => {
          state.error = `Transcription failed: ${err}`;
          state.isProcessing = false;
        });
      }
    },

    // Transcribe audio data (file bytes)
    transcribeAudio: async (audioData: ArrayBuffer) => {
      set((state) => {
        state.isProcessing = true;
        state.error = null;
      });

      try {
        const { sourceLanguage, targetLanguage } = get();
        const audioBytes = Array.from(new Uint8Array(audioData));

        const result = await invoke<TranscriptionResult>("transcribe_audio", {
          audioBytes,
          params: {
            source_language: sourceLanguage,
            target_language: targetLanguage,
          },
        });

        set((state) => {
          state.transcript = result.text;
          state.isProcessing = false;
        });
      } catch (err) {
        set((state) => {
          state.error = `Transcription failed: ${err}`;
          state.isProcessing = false;
        });
      }
    },

    // Transcribe a file
    transcribeFile: async (file: File) => {
      set((state) => {
        state.isProcessing = true;
        state.error = null;
      });

      try {
        const { sourceLanguage, targetLanguage } = get();
        const arrayBuffer = await file.arrayBuffer();
        const audioBytes = Array.from(new Uint8Array(arrayBuffer));

        const result = await invoke<TranscriptionResult>("transcribe_audio", {
          audioBytes,
          params: {
            source_language: sourceLanguage,
            target_language: targetLanguage,
          },
        });

        set((state) => {
          state.transcript = result.text;
          state.isProcessing = false;
        });
      } catch (err) {
        set((state) => {
          state.error = `Transcription failed: ${err}`;
          state.isProcessing = false;
        });
      }
    },

    // Clear transcript
    clearTranscript: () => {
      set((state) => {
        state.transcript = "";
      });
    },

    // Clear error
    clearError: () => {
      set((state) => {
        state.error = null;
      });
    },

    // Set recording duration
    setRecordingDuration: (duration: number) => {
      set((state) => {
        state.recordingDuration = duration;
      });
    },
  }))
);
