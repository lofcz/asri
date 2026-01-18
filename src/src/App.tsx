import { useEffect, useRef } from "react";
import { useASRStore } from "./store/asr";
import {
  RecordButton,
  FileDropzone,
  TranscriptDisplay,
  StatusIndicator,
  ModelSettings,
} from "./components";
import "./App.css";

function App() {
  const { fetchAvailableModels, isModelLoaded, modelStatus } = useASRStore();
  const initAttempted = useRef(false);

  // Fetch available models on mount
  useEffect(() => {
    if (!initAttempted.current) {
      initAttempted.current = true;
      fetchAvailableModels();
    }
  }, [fetchAvailableModels]);

  // Get model display name for footer
  const getModelFooterInfo = () => {
    if (!modelStatus?.is_loaded) return null;
    
    const modelNames: Record<string, string> = {
      parakeet: "NVIDIA Parakeet TDT 0.6B v3",
      canary: "NVIDIA Canary 1B v2",
    };
    
    return modelNames[modelStatus.model_type || ""] || modelStatus.model_type;
  };

  const modelFooterInfo = getModelFooterInfo();

  return (
    <div className="min-h-screen bg-surface relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Gradient orbs */}
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-accent/20 rounded-full blur-3xl" />
        
        {/* Grid pattern */}
        <div 
          className="absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                             linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
            backgroundSize: "50px 50px",
          }}
        />
      </div>

      {/* Main content */}
      <div className="relative z-10 container mx-auto px-6 py-12 max-w-4xl">
        {/* Header */}
        <header className="text-center mb-8">
          <div className="inline-flex items-center gap-3 mb-4">
            {/* Logo */}
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20">
              <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
              </svg>
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-100 to-slate-300 bg-clip-text text-transparent">
              ASRI
            </h1>
          </div>
          <p className="text-slate-400 text-lg">
            Automatic Speech Recognition{" "}
            {modelStatus?.model_type === "canary" && "& Translation "}
            powered by{" "}
            <span className="text-primary font-medium">NeMo</span>
          </p>
        </header>

        {/* Model Settings */}
        <div className="mb-6">
          <ModelSettings />
        </div>

        {/* Status indicator */}
        <div className="mb-8">
          <StatusIndicator />
        </div>

        {/* Main interaction area - only show when model is loaded */}
        {isModelLoaded ? (
          <>
            <div className="grid gap-8 lg:grid-cols-2 mb-8">
              {/* Left: Record button */}
              <div className="flex flex-col items-center justify-center p-8 rounded-3xl bg-gradient-to-br from-surface-light/80 to-surface-light/40 border border-slate-700/50 backdrop-blur-sm">
                <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-6">
                  Record Audio
                </h2>
                <RecordButton />
              </div>

              {/* Right: File upload */}
              <div className="flex flex-col p-8 rounded-3xl bg-gradient-to-br from-surface-light/80 to-surface-light/40 border border-slate-700/50 backdrop-blur-sm">
                <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-6">
                  Upload File
                </h2>
                <div className="flex-1 flex items-center">
                  <div className="w-full">
                    <FileDropzone />
                  </div>
                </div>
              </div>
            </div>

            {/* Transcript display */}
            <TranscriptDisplay />
          </>
        ) : (
          <div className="text-center py-16 rounded-3xl bg-gradient-to-br from-surface-light/40 to-surface-light/20 border border-slate-700/30">
            <svg className="w-16 h-16 mx-auto text-slate-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
            </svg>
            <h3 className="text-lg font-medium text-slate-400 mb-2">No Model Loaded</h3>
            <p className="text-sm text-slate-500 max-w-md mx-auto">
              Select a model and configuration above, then click "Load Model" to begin transcribing audio.
            </p>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 text-center">
          <p className="text-xs text-slate-600">
            {modelFooterInfo ? (
              <>
                Powered by{" "}
                <a
                  href={modelStatus?.model_type === "canary" 
                    ? "https://huggingface.co/nvidia/canary-1b-v2"
                    : "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"
                  }
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-500 hover:text-primary transition-colors"
                >
                  {modelFooterInfo}
                </a>
              </>
            ) : (
              "Select a model to get started"
            )}
            {" "}• Built with{" "}
            <a
              href="https://tauri.app"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-primary transition-colors"
            >
              Tauri
            </a>
            {" "}+{" "}
            <a
              href="https://ort.pyke.io"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-primary transition-colors"
            >
              ort
            </a>
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
