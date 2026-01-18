import { useState } from "react";
import { useASRStore } from "../store/asr";

export function TranscriptDisplay() {
  const { transcript, clearTranscript, isProcessing } = useASRStore();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!transcript) return;
    
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-medium text-slate-400 uppercase tracking-wider">
          Transcript
        </h2>
        
        <div className="flex items-center gap-2">
          {transcript && (
            <>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg
                  bg-slate-700/50 text-slate-300 hover:bg-slate-700 hover:text-white
                  transition-colors"
              >
                {copied ? (
                  <>
                    <svg className="w-3.5 h-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Copied!
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    Copy
                  </>
                )}
              </button>

              <button
                onClick={clearTranscript}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg
                  bg-slate-700/50 text-slate-300 hover:bg-red-500/20 hover:text-red-400
                  transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Clear
              </button>
            </>
          )}
        </div>
      </div>

      {/* Transcript area */}
      <div
        className={`
          relative min-h-[200px] max-h-[400px] overflow-y-auto rounded-2xl p-6
          bg-gradient-to-br from-surface-light to-surface border border-slate-700/50
          ${!transcript && !isProcessing ? "flex items-center justify-center" : ""}
        `}
      >
        {isProcessing ? (
          <div className="flex items-center justify-center h-full min-h-[150px]">
            <div className="flex flex-col items-center gap-4">
              {/* Animated waveform */}
              <div className="flex items-center gap-1 h-12">
                {[...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className="w-1.5 bg-gradient-to-t from-primary to-accent rounded-full animate-pulse"
                    style={{
                      height: `${20 + Math.random() * 30}px`,
                      animationDelay: `${i * 0.1}s`,
                      animationDuration: "0.6s",
                    }}
                  />
                ))}
              </div>
              <p className="text-slate-400 text-sm">Transcribing audio...</p>
            </div>
          </div>
        ) : transcript ? (
          <p className="font-mono text-lg leading-relaxed text-slate-100 whitespace-pre-wrap">
            {transcript}
          </p>
        ) : (
          <div className="text-center text-slate-500">
            <svg
              className="w-12 h-12 mx-auto mb-3 opacity-50"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <p>Your transcription will appear here</p>
          </div>
        )}

        {/* Gradient fade at bottom when scrollable */}
        {transcript && (
          <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-surface-light to-transparent pointer-events-none rounded-b-2xl" />
        )}
      </div>
    </div>
  );
}
