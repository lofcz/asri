import { useEffect, useState } from "react";
import { useASRStore } from "../store/asr";

export function ModelSettings() {
  const {
    availableModels,
    selectedModelId,
    selectedQuantization,
    sourceLanguage,
    targetLanguage,
    isModelLoaded,
    isModelLoading,
    fetchAvailableModels,
    setSelectedModel,
    setSelectedQuantization,
    setSourceLanguage,
    setTargetLanguage,
    initializeModel,
    modelStatus,
  } = useASRStore();

  const [isExpanded, setIsExpanded] = useState(false);

  // Fetch available models on mount
  useEffect(() => {
    fetchAvailableModels();
  }, [fetchAvailableModels]);

  const selectedModel = availableModels.find((m) => m.id === selectedModelId);
  const supportsTranslation = selectedModel?.supports_languages ?? false;

  const handleLoadModel = () => {
    initializeModel();
  };

  // Get display name for current config
  const getConfigSummary = () => {
    if (modelStatus?.is_loaded) {
      const modelName = availableModels.find(m => m.id === modelStatus.model_type)?.name || modelStatus.model_type;
      const quant = modelStatus.quantization === "int8" ? "INT8" : "FP32";
      return `${modelName} (${quant})`;
    }
    return "No model loaded";
  };

  return (
    <div className="rounded-2xl bg-surface-light/60 border border-slate-700/50 overflow-hidden">
      {/* Header - always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-5 py-3.5 flex items-center justify-between hover:bg-slate-700/20 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${isModelLoaded ? "bg-emerald-400" : isModelLoading ? "bg-amber-400 animate-pulse" : "bg-slate-500"}`} />
          <div className="text-left">
            <span className="text-sm font-medium text-slate-200">Model</span>
            <span className="text-sm text-slate-400 ml-2">{getConfigSummary()}</span>
          </div>
        </div>
        <svg
          className={`w-5 h-5 text-slate-400 transition-transform ${isExpanded ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded settings panel */}
      {isExpanded && (
        <div className="px-5 pb-5 border-t border-slate-700/50">
          <div className="pt-4 space-y-5">
            {/* Model Selection */}
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                Model
              </label>
              <div className="grid grid-cols-2 gap-2">
                {availableModels.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setSelectedModel(model.id)}
                    disabled={isModelLoading}
                    className={`
                      p-3 rounded-xl text-left transition-all border
                      ${selectedModelId === model.id
                        ? "bg-primary/20 border-primary/50 text-slate-100"
                        : "bg-slate-800/50 border-slate-700/50 text-slate-300 hover:border-slate-600"
                      }
                      ${isModelLoading ? "opacity-50 cursor-not-allowed" : ""}
                    `}
                  >
                    <div className="font-medium text-sm">{model.name}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{model.description}</div>
                    {model.supports_languages && (
                      <div className="flex items-center gap-1 mt-1.5">
                        <svg className="w-3.5 h-3.5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                        </svg>
                        <span className="text-xs text-accent">{model.supported_languages.length} languages</span>
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Quantization Selection */}
            <div className="space-y-2">
              <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                Precision
              </label>
              <div className="flex gap-2">
                {selectedModel?.quantizations.map((quant) => (
                  <button
                    key={quant.id}
                    onClick={() => setSelectedQuantization(quant.id)}
                    disabled={isModelLoading}
                    className={`
                      flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all border
                      ${selectedQuantization === quant.id
                        ? "bg-primary/20 border-primary/50 text-slate-100"
                        : "bg-slate-800/50 border-slate-700/50 text-slate-300 hover:border-slate-600"
                      }
                      ${isModelLoading ? "opacity-50 cursor-not-allowed" : ""}
                    `}
                  >
                    {quant.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Language Selection (only for multilingual models) */}
            {supportsTranslation && (
              <div className="space-y-3">
                <div className="space-y-2">
                  <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                    Source Language
                  </label>
                  <select
                    value={sourceLanguage}
                    onChange={(e) => setSourceLanguage(e.target.value)}
                    disabled={isModelLoading}
                    className="w-full py-2.5 px-3 rounded-lg bg-slate-800/70 border border-slate-700/50 text-slate-200 text-sm focus:outline-none focus:border-primary/50 disabled:opacity-50"
                  >
                    {selectedModel?.supported_languages.map((lang) => (
                      <option key={lang.code} value={lang.code}>
                        {lang.name} ({lang.code})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="block text-xs font-medium text-slate-400 uppercase tracking-wider">
                      Target Language
                    </label>
                    <span className="text-xs text-slate-500">(for translation)</span>
                  </div>
                  <select
                    value={targetLanguage || sourceLanguage}
                    onChange={(e) => setTargetLanguage(e.target.value === sourceLanguage ? null : e.target.value)}
                    disabled={isModelLoading}
                    className="w-full py-2.5 px-3 rounded-lg bg-slate-800/70 border border-slate-700/50 text-slate-200 text-sm focus:outline-none focus:border-primary/50 disabled:opacity-50"
                  >
                    {selectedModel?.supported_languages.map((lang) => (
                      <option key={lang.code} value={lang.code}>
                        {lang.name} ({lang.code})
                        {lang.code === sourceLanguage ? " — same as source" : ""}
                      </option>
                    ))}
                  </select>
                  {targetLanguage && targetLanguage !== sourceLanguage && (
                    <p className="text-xs text-accent flex items-center gap-1">
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                      Will translate {sourceLanguage.toUpperCase()} → {targetLanguage.toUpperCase()}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Load Button */}
            <button
              onClick={handleLoadModel}
              disabled={isModelLoading}
              className={`
                w-full py-3 px-4 rounded-xl font-medium text-sm transition-all
                ${isModelLoading
                  ? "bg-slate-700 text-slate-400 cursor-wait"
                  : isModelLoaded
                    ? "bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-600/30"
                    : "bg-primary text-white hover:bg-primary/90 shadow-lg shadow-primary/20"
                }
              `}
            >
              {isModelLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Loading Model...
                </span>
              ) : isModelLoaded ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Model Loaded — Reload
                </span>
              ) : (
                "Load Model"
              )}
            </button>

            {!isModelLoaded && !isModelLoading && (
              <p className="text-xs text-center text-slate-500">
                Select your model and precision, then click Load to initialize
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
