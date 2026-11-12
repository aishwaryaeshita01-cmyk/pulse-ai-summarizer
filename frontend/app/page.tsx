"use client";

import React, { useState } from "react";
import { Upload, FileAudio, CheckCircle2, AlertCircle, Loader2, Sparkles } from "lucide-react";

interface StructuredSummary {
  executive_summary: string;
  key_takeaways: string[];
  action_items: string[];
  sentiment: string;
}

interface PipelineResponse {
  success: boolean;
  transcription_model: string;
  summarization_model: string;
  transcript: string;
  analysis: StructuredSummary;
}

export default function PulseDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [tier, setTier] = useState<string>("high_tier_reasoning");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PipelineResponse | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("tier", tier);

    try {
      const response = await fetch("http://localhost:8000/api/v1/process-audio", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Pipeline execution failed.");
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message || "Unable to reach backend server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <header className="border-b border-slate-800 pb-6">
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Sparkles className="text-indigo-400" /> Pulse AI Audio Intelligence
          </h1>
          <p className="text-slate-400 mt-1">
            Resilient Speech-to-Text & Schema-Validated Summarization Pipeline
          </p>
        </header>

        {/* Upload Form */}
        <form onSubmit={handleSubmit} className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
          <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-indigo-500 transition">
            <input
              type="file"
              accept=".mp3,.wav,.m4a,.webm,.ogg,.flac"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="hidden"
              id="audio-upload"
            />
            <label htmlFor="audio-upload" className="cursor-pointer flex flex-col items-center gap-2">
              {file ? (
                <>
                  <FileAudio className="w-10 h-10 text-indigo-400" />
                  <span className="font-medium text-indigo-300">{file.name}</span>
                  <span className="text-xs text-slate-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </>
              ) : (
                <>
                  <Upload className="w-10 h-10 text-slate-400" />
                  <span className="font-medium">Click to select an audio file</span>
                  <span className="text-xs text-slate-500">MP3, WAV, M4A, WEBM, OGG, FLAC (Up to 100MB)</span>
                </>
              )}
            </label>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-between items-center">
            <select
              value={tier}
              onChange={(e) => setTier(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-sm w-full sm:w-auto"
            >
              <option value="high_tier_reasoning">High-Tier Reasoning (GPT-OSS 120B)</option>
              <option value="fast_summary">Fast Summary (Qwen 3.6 27B)</option>
            </select>

            <button
              type="submit"
              disabled={!file || loading}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:cursor-not-allowed px-6 py-2.5 rounded-lg font-medium flex items-center gap-2 w-full sm:w-auto justify-center transition"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              {loading ? "Processing Pipeline..." : "Analyze Audio"}
            </button>
          </div>
        </form>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-950/50 border border-red-800 text-red-200 p-4 rounded-xl flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Results Dashboard */}
        {result && (
          <section className="space-y-6">
            {/* Telemetry Badges */}
            <div className="flex flex-wrap gap-3 text-xs">
              <span className="bg-emerald-950/60 border border-emerald-800 text-emerald-300 px-3 py-1 rounded-full flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" /> STT: {result.transcription_model}
              </span>
              <span className="bg-indigo-950/60 border border-indigo-800 text-indigo-300 px-3 py-1 rounded-full">
                LLM: {result.summarization_model}
              </span>
              <span className="bg-slate-800 border border-slate-700 text-slate-300 px-3 py-1 rounded-full">
                Sentiment: {result.analysis.sentiment}
              </span>
            </div>

            {/* Executive Summary */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-indigo-400 mb-2">Executive Summary</h2>
              <p className="text-slate-300 leading-relaxed">{result.analysis.executive_summary}</p>
            </div>

            {/* Takeaways & Action Items */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h3 className="text-md font-semibold text-slate-200 mb-3">Key Takeaways</h3>
                <ul className="space-y-2 text-sm text-slate-300 list-disc pl-5">
                  {result.analysis.key_takeaways.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h3 className="text-md font-semibold text-slate-200 mb-3">Action Items</h3>
                {result.analysis.action_items.length > 0 ? (
                  <ul className="space-y-2 text-sm text-slate-300 list-disc pl-5">
                    {result.analysis.action_items.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-slate-500">No explicit action items detected.</p>
                )}
              </div>
            </div>

            {/* Full Transcript */}
            <details className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <summary className="cursor-pointer font-medium text-slate-300">View Full Raw Transcript</summary>
              <p className="mt-4 text-sm text-slate-400 whitespace-pre-wrap leading-relaxed">
                {result.transcript}
              </p>
            </details>
          </section>
        )}
      </div>
    </main>
  );
}
