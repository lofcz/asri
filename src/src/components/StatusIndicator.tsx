import { useASRStore } from "../store/asr";

export function StatusIndicator() {
  const { isModelLoaded, isModelLoading, error, clearError } = useASRStore();

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Model status */}
      <div className="flex items-center gap-2">
        <div
          className={`
            w-2.5 h-2.5 rounded-full transition-colors
            ${isModelLoading 
              ? "bg-yellow-400 animate-pulse" 
              : isModelLoaded 
                ? "bg-green-400" 
                : "bg-red-400"
            }
          `}
        />
        <span className="text-sm text-slate-400">
          {isModelLoading
            ? "Loading model..."
            : isModelLoaded
              ? "Model ready"
              : "Model not loaded"}
        </span>
      </div>

      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/20">
          <svg
            className="w-4 h-4 text-red-400 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span className="text-sm text-red-400">{error}</span>
          <button
            onClick={clearError}
            className="ml-2 p-1 rounded hover:bg-red-500/20 transition-colors"
          >
            <svg className="w-3.5 h-3.5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      {/* Loading spinner when model is loading */}
      {isModelLoading && (
        <div className="flex items-center gap-3 mt-2">
          <div className="relative w-12 h-12">
            <svg className="w-12 h-12 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle
                className="opacity-20"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="2"
              />
              <path
                className="text-primary"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          </div>
          <div className="text-left">
            <p className="text-sm font-medium text-slate-200">Initializing ASR Engine</p>
            <p className="text-xs text-slate-500">Loading ONNX models...</p>
          </div>
        </div>
      )}
    </div>
  );
}
