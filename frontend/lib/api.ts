export type Severity = "Low" | "Medium" | "High";
export type Confidence = "Low" | "Medium" | "High";

export interface DetectedPattern {
  pattern: string;
  severity: Severity;
  score: number;
  examples: string[];
}

export interface SentenceScore {
  index: number;
  text: string;
  ai_likelihood: number;
}

export interface Suggestion {
  detected_sentence: string | null;
  issue: string;
  suggestion: string;
  improved_direction: string | null;
}

export interface OverallResult {
  ai_writing_likelihood: number;
  confidence: Confidence;
}

export interface AnalyzeResponse {
  overall: OverallResult;
  detected_patterns: DetectedPattern[];
  sentence_scores: SentenceScore[];
  highlighted_phrases: string[];
  suggestions: Suggestion[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export async function analyzeContent(
  content: string,
  includeSuggestions = true
): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, include_suggestions: includeSuggestions }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Analysis failed (${res.status}): ${detail || res.statusText}`);
  }

  return res.json();
}
