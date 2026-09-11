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

function getApiBase(): string {
  if (typeof window !== "undefined" && window.location?.hostname) {
    const protocol = window.location.protocol || "http:";
    const hostname = window.location.hostname;
    return `${protocol}//${hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
}

export async function analyzeContent(
  content: string,
  includeSuggestions = true,
  userId?: number
): Promise<AnalyzeResponse> {
  const res = await fetch(`${getApiBase()}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      include_suggestions: includeSuggestions,
      user_id: userId ?? null,
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Analysis failed (${res.status}): ${detail || res.statusText}`);
  }

  return res.json();
}

export interface User {
  id: number;
  name: string;
  created_at: string;
}

export async function listUsers(): Promise<User[]> {
  const res = await fetch(`${getApiBase()}/users`);
  if (!res.ok) {
    throw new Error(`Failed to load users (${res.status})`);
  }
  return res.json();
}

export async function createUser(name: string): Promise<User> {
  const res = await fetch(`${getApiBase()}/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not create user (${res.status}): ${detail || res.statusText}`);
  }
  return res.json();
}

export async function deleteUser(id: number): Promise<void> {
  const res = await fetch(`${getApiBase()}/users/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not delete user (${res.status}): ${detail || res.statusText}`);
  }
}

export interface Profile {
  id: number;
  name: string;
  description: string;
  phrase_count: number;
  created_at: string;
}

export async function listProfiles(userId?: number): Promise<Profile[]> {
  const url = userId !== undefined ? `${getApiBase()}/profiles?user_id=${userId}` : `${getApiBase()}/profiles`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load profiles (${res.status})`);
  }
  return res.json();
}

export async function createProfile(name: string, description = "", userId?: number): Promise<Profile> {
  const res = await fetch(`${getApiBase()}/profiles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description, user_id: userId ?? null }),
  });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not create profile (${res.status}): ${detail || res.statusText}`);
  }
  return res.json();
}

export async function updateProfile(id: number, name: string, description = ""): Promise<Profile> {
  const res = await fetch(`${getApiBase()}/profiles/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not update profile (${res.status}): ${detail || res.statusText}`);
  }
  return res.json();
}

export async function deleteProfile(id: number): Promise<void> {
  const res = await fetch(`${getApiBase()}/profiles/${id}`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not delete profile (${res.status}): ${detail || res.statusText}`);
  }
}

export interface TrainedPhrase {
  id: number;
  ai_phrase: string;
  humanized_phrase: string;
  profile_id?: number;
  created_at: string;
}

export async function listTrainedPhrases(profileId?: number): Promise<TrainedPhrase[]> {
  const url = profileId !== undefined ? `${getApiBase()}/train/phrases?profile_id=${profileId}` : `${getApiBase()}/train/phrases`;
  const res = await fetch(url);
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
  humanizedPhrase: string,
  profileId?: number
): Promise<TrainedPhrase> {
  const res = await fetch(`${getApiBase()}/train/phrases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ai_phrase: aiPhrase,
      humanized_phrase: humanizedPhrase,
      profile_id: profileId ?? null,
    }),
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
  humanizedPhrase: string,
  profileId?: number
): Promise<TrainedPhrase> {
  const res = await fetch(`${getApiBase()}/train/phrases/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ai_phrase: aiPhrase,
      humanized_phrase: humanizedPhrase,
      profile_id: profileId ?? null,
    }),
  });
  if (!res.ok) {
    const detail = await parseErrorDetail(res);
    throw new Error(`Could not update phrase (${res.status}): ${detail || res.statusText}`);
  }
  return res.json();
}

export async function deleteTrainedPhrase(id: number): Promise<void> {
  const res = await fetch(`${getApiBase()}/train/phrases/${id}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Could not delete phrase (${res.status})`);
  }
}

export interface HumanizeResponse {
  humanized_content: string;
  ai_score_after?: number | null;
}

export async function humanizeContent(
  content: string,
  projectId?: number,
  profileId?: number,
  phraseIds?: number[]
): Promise<HumanizeResponse> {
  const res = await fetch(`${getApiBase()}/humanize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      project_id: projectId ?? null,
      profile_id: profileId ?? null,
      phrase_ids: phraseIds && phraseIds.length > 0 ? phraseIds : null,
    }),
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

export async function listProjects(userId?: number): Promise<ProjectSummary[]> {
  const url = userId !== undefined ? `${getApiBase()}/projects?user_id=${userId}` : `${getApiBase()}/projects`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load history (${res.status})`);
  }
  return res.json();
}

export async function getProject(id: number): Promise<ProjectDetail> {
  const res = await fetch(`${getApiBase()}/projects/${id}`);
  if (!res.ok) {
    throw new Error(`Failed to load project (${res.status})`);
  }
  return res.json();
}

export async function deleteProject(id: number): Promise<void> {
  const res = await fetch(`${getApiBase()}/projects/${id}`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Could not delete project (${res.status})`);
  }
}
