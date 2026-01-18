import { useASRStore } from "../store/asr";

export function RecordButton() {
  const { isRecording, isProcessing, isModelLoaded, startRecording, stopRecording, recordingDuration } =
    useASRStore();

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const handleClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const isDisabled = !isModelLoaded || isProcessing;

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Main record button */}
      <button
        onClick={handleClick}
        disabled={isDisabled}
        className={`
          relative w-24 h-24 rounded-full transition-all duration-300 ease-out
          flex items-center justify-center
          ${isDisabled 
            ? "bg-slate-700 cursor-not-allowed opacity-50" 
            : isRecording
              ? "bg-red-500 hover:bg-red-400 shadow-lg shadow-red-500/30"
              : "bg-gradient-to-br from-primary to-accent hover:shadow-lg hover:shadow-primary/30 hover:scale-105"
          }
          focus:outline-none focus:ring-4 focus:ring-primary/30
        `}
      >
        {/* Pulse animation when recording */}
        {isRecording && (
          <>
            <span className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-30" />
            <span className="absolute inset-2 rounded-full bg-red-400 animate-pulse opacity-40" />
          </>
        )}

        {/* Icon */}
        <svg
          className={`w-10 h-10 transition-transform ${isRecording ? "scale-90" : ""}`}
          fill="currentColor"
          viewBox="0 0 24 24"
        >
          {isRecording ? (
            // Stop icon
            <rect x="6" y="6" width="12" height="12" rx="2" />
          ) : (
            // Microphone icon
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z" />
          )}
        </svg>
      </button>

      {/* Duration display */}
      <div
        className={`
          text-sm font-mono transition-opacity duration-200
          ${isRecording ? "text-red-400" : "text-slate-500"}
        `}
      >
        {isRecording ? (
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            {formatDuration(recordingDuration)}
          </span>
        ) : (
          <span>Push to talk</span>
        )}
      </div>

      {/* Processing indicator */}
      {isProcessing && (
        <div className="flex items-center gap-2 text-primary text-sm">
          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          Processing...
        </div>
      )}
    </div>
  );
}
