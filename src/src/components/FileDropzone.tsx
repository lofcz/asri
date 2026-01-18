import { useCallback, useState } from "react";
import { useASRStore } from "../store/asr";

export function FileDropzone() {
  const { transcribeFile, isProcessing, isModelLoaded } = useASRStore();
  const [isDragging, setIsDragging] = useState(false);

  const supportedFormats = ["wav", "mp3", "flac", "ogg", "m4a", "aac", "webm"];

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      if (!isModelLoaded || isProcessing) return;

      const files = Array.from(e.dataTransfer.files);
      const audioFile = files.find((f) => {
        const ext = f.name.split(".").pop()?.toLowerCase();
        return ext && supportedFormats.includes(ext);
      });

      if (audioFile) {
        transcribeFile(audioFile);
      }
    },
    [isModelLoaded, isProcessing, transcribeFile]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        transcribeFile(file);
      }
      // Reset input
      e.target.value = "";
    },
    [transcribeFile]
  );

  const isDisabled = !isModelLoaded || isProcessing;

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        relative rounded-2xl border-2 border-dashed p-8 transition-all duration-200
        ${isDisabled 
          ? "border-slate-700 bg-slate-800/30 cursor-not-allowed opacity-50" 
          : isDragging
            ? "border-primary bg-primary/10 scale-[1.02]"
            : "border-slate-600 bg-surface-light/50 hover:border-slate-500 hover:bg-surface-light"
        }
      `}
    >
      <input
        type="file"
        accept={supportedFormats.map((f) => `.${f}`).join(",")}
        onChange={handleFileSelect}
        disabled={isDisabled}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
      />

      <div className="flex flex-col items-center gap-4 pointer-events-none">
        {/* Upload icon */}
        <div
          className={`
            w-16 h-16 rounded-full flex items-center justify-center transition-colors
            ${isDragging ? "bg-primary/20 text-primary" : "bg-slate-700/50 text-slate-400"}
          `}
        >
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>

        {/* Text */}
        <div className="text-center">
          <p className={`text-lg font-medium ${isDragging ? "text-primary" : "text-slate-200"}`}>
            {isDragging ? "Drop audio file here" : "Drop audio file or click to browse"}
          </p>
          <p className="text-sm text-slate-500 mt-1">
            Supports {supportedFormats.slice(0, -1).join(", ")}, and {supportedFormats.slice(-1)}
          </p>
        </div>
      </div>
    </div>
  );
}
