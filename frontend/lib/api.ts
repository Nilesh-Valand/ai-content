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
  id: number;
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

export interface TrainedPhrase {
  id: number;
  ai_phrase: string;
  humanized_phrase: string;
  created_at: string;
}

export async function listTrainedPhrases(): Promise<TrainedPhrase[]> {
  const res = await fetch(`${API_BASE}/train/phrases`);
  if (!res.ok) {
    throw new Error(`Failed to load trained phrases (${res.status})`);
  }
  return res.json();
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return Array.isArray(body.detail)
      ? body.detail.map((d: { msg?: string }) => d.msg).join(", ")
      : body.detail || "";
  } catch {
    return await res.text().catch(() => "");
  }
}

export async function createTrainedPhrase(
  aiPhrase: string,
  humanizedPhrase: string
): Promise<TrainedPhrase> {
  const res = await fetch(`${API_BASE}/train/phrases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ai_phrase: aiPhrase, humanized_phrase: humanizedPhrase }),
  });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not save phrase (${res.status}): ${detail || res.statusText}`);
  }
  return res.json();
}

export async function updateTrainedPhrase(
  id: number,
  aiPhrase: string,
  humanizedPhrase: string
): Promise<TrainedPhrase> {
  const res = await fetch(`${API_BASE}/train/phrases/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ai_phrase: aiPhrase, humanized_phrase: humanizedPhrase }),
  });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not update phrase (${res.status}): ${detail || res.statusText}`);
  }
  return res.json();
}

export async function deleteTrainedPhrase(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/train/phrases/${id}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Could not delete phrase (${res.status})`);
  }
}

export interface HumanizeResponse {
  humanized_content: string;
}

export async function humanizeContent(
  content: string,
  projectId?: number
): Promise<HumanizeResponse> {
  const res = await fetch(`${API_BASE}/humanize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, project_id: projectId ?? null }),
  });

  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || "";
    } catch {
      detail = await res.text().catch(() => "");
    }
    throw new Error(`Humanization failed (${res.status}): ${detail || res.statusText}`);
  }

  return res.json();
}

export interface ProjectSummary {
  id: number;
  content: string;
  ai_writing_likelihood: number;
  confidence: Confidence;
  has_humanized: boolean;
  created_at: string;
}

export interface ProjectDetail {
  id: number;
  content: string;
  overall: OverallResult;
  detected_patterns: DetectedPattern[];
  sentence_scores: SentenceScore[];
  highlighted_phrases: string[];
  suggestions: Suggestion[];
  humanized_content: string | null;
  created_at: string;
}

export async function listProjects(): Promise<ProjectSummary[]> {
  const res = await fetch(`${API_BASE}/projects`);
  if (!res.ok) {
    throw new Error(`Failed to load history (${res.status})`);
  }
  return res.json();
}

export async function getProject(id: number): Promise<ProjectDetail> {
  const res = await fetch(`${API_BASE}/projects/${id}`);
  if (!res.ok) {
    throw new Error(`Failed to load project (${res.status})`);
  }
  return res.json();
}

export async function deleteProject(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/projects/${id}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Could not delete project (${res.status})`);
  }
}
