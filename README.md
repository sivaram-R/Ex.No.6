# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Automating Multi-AI Tool Interaction, Comparison, and Insight Generatio

# 1. Executive summary

This document presents a complete approach and code to interact with multiple AI tool APIs, collect outputs for given prompts, compare those outputs quantitatively and qualitatively, and produce actionable insights automatically. The approach uses a *persona pattern* — i.e., the orchestrator can ask different AI systems to respond in specific personas (e.g., `developer`, `analyst`, `critic`) to induce different styles and viewpoints. The included Python code is modular and ready to run after replacing API keys and installing dependencies.

---

# 2. Goals & scope

* Integrate with **multiple AI models/APIs** (examples: OpenAI, Hugging Face Inference, Anthropic, local LLM) via modular client wrappers.
* Allow prompt templating plus persona injection.
* Run batches of prompts across models and personas, collect outputs.
* Compute automated comparisons using objective metrics (semantic similarity, ROUGE, BLEU, length, lexical diversity, sentiment).
* Produce a human-readable report and machine-readable JSON with conclusions and actionable recommendations.
* Be reproducible and extensible: add/remove models easily.

Scope excludes training/finetuning models (only inference), and does not require paid endpoints to be actually called — placeholders are used. Replace with real credentials to run.

---

# 3. High-level architecture

1. **Input layer**: Prompt templates, dataset (list of tasks/questions), persona definitions.
2. **Orchestrator**: Loops through prompts × personas × models, collects responses, logs metadata (latency, tokens).
3. **Comparison Engine**: Computes metrics between outputs (e.g., pairwise cosine similarity of embeddings, ROUGE/L, lexical metrics).
4. **Insight Generator**: Applies heuristics and thresholds to produce recommendations (e.g., “Model A is more concise and factual; Model B is more creative; prefer A for summaries”).
5. **Reporter**: Exports a PDF/HTML/Markdown and JSON with results and visualization-ready data.

---

# 4. Persona pattern & why it helps

A *persona* is an instruction prefix that conditions a model to adopt a style, level, or role. Example personas:

* `developer`: Focus on code, implementation steps, brevity, accuracy.
* `manager`: Focus on high-level outcomes, risks, timelines.
* `critic`: Provide pros/cons, points of failure.
* `teacher`: Explain step-by-step for beginners.

Why use personas?

* Elicits varied perspectives from the same model — increases robustness of evaluation.
* Helps discover what role each model performs best in.
* Useful when generating multi-stakeholder output (e.g., one doc for devs, one for managers).

---

# 5. Choice of AI tools (examples)

You can adapt to whatever APIs you have. The code demonstrates the following wrappers:

* **OpenAI** (ChatCompletion / responses / embeddings) — widely used.
* **Hugging Face Inference API** — for models not on OpenAI.
* **Anthropic Claude** (example wrapper — token and endpoints similar) — provides alternative safety/response patterns.
* **Local LLM** via `llama_cpp` or HTTP wrapper — optional.

The code is modular so you can drop a wrapper or add a new one quickly.

---

# 6. Experiment design & evaluation metrics

**Dataset**: a set of prompts/tasks you want to compare across models (e.g., summarization tasks, code generation tasks, question answering).

**Metrics to compute automatically**

* **Semantic similarity**: cosine similarity between sentence embeddings to detect agreement. (Using sentence-transformers or embeddings from model APIs.)
* **ROUGE-L / ROUGE-1 / ROUGE-2**: for summarization/QA when reference exists.
* **BLEU**: for exactness (code samples rarely benefit).
* **BERTScore**: semantic overlap measured on token embeddings.
* **Factuality proxy**: simple heuristics (presence of numbers, dates, named entities); optionally external verification step (web.run) — not included here.
* **Readability**: Flesch reading ease / grade level.
* **Length & verbosity**: token/word counts.
* **Sentiment / toxicity**: quick check with pretrained classifier.

**Comparison strategy**

* Pairwise model comparisons for each prompt.
* Aggregate model-level statistics across all prompts.
* Persona-level analysis: which persona produced more actionable steps, more pros/cons, etc.

**Decision rules for insights (examples)**

* If semantic similarity < 0.6 between model A and B → mark as "high disagreement".
* If one model consistently produces shorter, more factual answers (based on heuristics) → recommend for executive summaries.
* If a model produces high lexical diversity + low similarity to reference → recommend for creative tasks.

---

# 7. Implementation (full Python code + explanations)

> **Important**: Replace `YOUR_OPENAI_KEY`, `YOUR_HF_KEY`, `YOUR_ANTHROPIC_KEY` with your secrets. Never commit keys.

## 7.1 Environment & dependencies

Install required packages:

```bash
pip install openai requests sentence-transformers transformers rouge-score sacrebleu bert_score textstat numpy pandas tqdm matplotlib
```

(You may not need all; adjust as needed.)

## 7.2 Code layout

We'll paste full code as modular single-file for convenience. Save as `multi_ai_orchestrator.py`.

---

## 7.3 Full Python implementation (copy-paste ready)

```python
"""
multi_ai_orchestrator.py

A single-file orchestrator that:
- Calls multiple AI APIs (OpenAI, HuggingFace, Anthropic example)
- Applies persona templates
- Collects outputs, embeddings
- Computes comparison metrics
- Produces a summary report (CSV/JSON) and prints insights.

Replace API keys and tune endpoints as necessary.
"""

import os
import time
import json
import math
from typing import List, Dict, Any
import requests
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np
import pandas as pd

# NLP / metrics
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore
import textstat

# -------------------------
# Configuration - set your keys here
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_KEY"
HF_API_KEY = os.getenv("HF_API_KEY") or "YOUR_HF_KEY"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or "YOUR_ANTHROPIC_KEY"

# For embeddings (uses sentence-transformers locally to avoid extra API calls)
EMBED_MODEL = "all-MiniLM-L6-v2"

# -------------------------
# Utilities
# -------------------------
def now_ts():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

# -------------------------
# Lightweight client wrappers
# -------------------------
# NOTE: These wrappers are minimal and for demonstration. For production, add retries, rate-limit handling, and robust error checks.

# 1) OpenAI chat + embeddings (example)
class OpenAIClient:
    def __init__(self, api_key=OPENAI_API_KEY, model="gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base = "https://api.openai.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def chat(self, prompt: str, system: str = None, max_tokens=512) -> Dict[str, Any]:
        url = f"{self.base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": []
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": prompt})
        payload["max_tokens"] = max_tokens
        r = requests.post(url, json=payload, headers=self.headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        # adapt to response shape
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "meta": data}

    # placeholder for embeddings via local model instead of API
    def embed(self, texts: List[str]):
        raise NotImplementedError("Use local embedding model (see Orchestrator)")

# 2) Hugging Face Inference API (text generation)
class HFClient:
    def __init__(self, api_key=HF_API_KEY, model="google/flan-t5-large"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://api-inference.huggingface.co/models/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def generate(self, prompt: str, max_length=256) -> Dict[str, Any]:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_length}}
        r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        elif isinstance(data, dict) and "error" in data:
            text = ""
        else:
            # Some models return raw text
            text = str(data)
        return {"text": text, "meta": data}

# 3) Anthropic (example call to Claude) - pseudocode
class AnthropicClient:
    def __init__(self, api_key=ANTHROPIC_API_KEY, model="claude-2"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.anthropic.com/v1/complete"
        self.headers = {"x-api-key": self.api_key}

    def complete(self, prompt: str, max_tokens=300):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens
        }
        r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return {"text": data.get("completion", ""), "meta": data}

# -------------------------
# Persona manager
# -------------------------
PERSONAS = {
    "developer": "You are an expert software engineer. Produce concise, code-focused answers with step-by-step instructions.",
    "manager": "You are a product manager. Focus on high-level outcomes, risks, and timelines in simple language.",
    "critic": "You are a critical reviewer. Provide pros, cons, pitfalls, and counterpoints.",
    "teacher": "You are a teacher explaining to a beginner. Use clear examples and step-by-step explanations."
}

# -------------------------
# Orchestrator
# -------------------------
@dataclass
class ResponseRecord:
    timestamp: str
    prompt_id: str
    prompt: str
    persona: str
    model_name: str
    model_provider: str
    text: str
    tokens: int = 0
    latency_s: float = 0.0

class Orchestrator:
    def __init__(self, openai_client: OpenAIClient = None, hf_client: HFClient = None, anth_client: AnthropicClient = None):
        self.openai = openai_client
        self.hf = hf_client
        self.anth = anth_client
        self.embedder = SentenceTransformer(EMBED_MODEL)  # local, fast
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def call_model(self, model_key: str, prompt: str, persona: str) -> ResponseRecord:
        system = PERSONAS.get(persona, "")
        start = time.time()
        if model_key == "openai":
            resp = self.openai.chat(prompt, system=system)
            txt = resp["text"]
            provider = "openai"
            model_name = self.openai.model
        elif model_key == "hf":
            # combine persona in prompt
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            resp = self.hf.generate(full_prompt)
            txt = resp["text"]
            provider = "huggingface"
            model_name = self.hf.model
        elif model_key == "anthropic":
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            resp = self.anth.complete(full_prompt)
            txt = resp["text"]
            provider = "anthropic"
            model_name = self.anth.model
        else:
            raise ValueError("Unknown model key")
        end = time.time()
        rec = ResponseRecord(
            timestamp=now_ts(),
            prompt_id=f"p-{int(time.time()*1000)}",
            prompt=prompt,
            persona=persona,
            model_name=model_name,
            model_provider=provider,
            text=txt.strip(),
            latency_s=end-start
        )
        return rec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def compute_pairwise_semantic(self, texts: List[str]) -> np.ndarray:
        emb = self.embed_texts(texts)
        # cosine similarity matrix
        sim = np.dot(emb, emb.T)
        return sim

    # metrics for a response (vs optional reference)
    def compute_metrics(self, response: str, reference: str = None) -> Dict[str, Any]:
        metrics = {}
        metrics["length_words"] = len(response.split())
        metrics["flesch_reading_ease"] = textstat.flesch_reading_ease(response)
        # ROUGE (if reference provided)
        if reference:
            scores = self.scorer.score(reference, response)
            metrics.update({k: v.fmeasure for k, v in scores.items()})
            # BLEU via sacrebleu
            bleu = sacrebleu.sentence_bleu(response, [reference]).score
            metrics["bleu"] = bleu
            # BERTScore
            P, R, F1 = bertscore([response], [reference], lang="en", rescale_with_baseline=True)
            metrics["bertscore_f1"] = float(F1[0])
        return metrics

    # orchestrate batch run
    def run_batch(self, prompts: List[str], models: List[str], personas: List[str], reference_texts: Dict[int, str] = None):
        records = []
        for i, prompt in enumerate(prompts):
            for persona in personas:
                for model_key in models:
                    try:
                        rec = self.call_model(model_key, prompt, persona)
                    except Exception as e:
                        print(f"Error calling {model_key}: {e}")
                        rec = ResponseRecord(timestamp=now_ts(), prompt_id=f"err-{i}", prompt=prompt, persona=persona,
                                             model_name=model_key, model_provider=model_key, text=f"ERROR: {e}", latency_s=0.0)
                    # compute some metrics
                    ref = None
                    if reference_texts and i in reference_texts:
                        ref = reference_texts[i]
                    m = self.compute_metrics(rec.text, ref)
                    rec_meta = asdict(rec)
                    rec_meta.update({"metrics": m})
                    records.append(rec_meta)
        return records

# -------------------------
# Insight generator
# -------------------------
def generate_insights(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(records)
    insights = {}
    # aggregate by model provider
    by_model = df.groupby("model_provider").agg(
        avg_length=("text", lambda t: np.mean([len(x.split()) for x in t])),
        avg_flesch=("metrics", lambda m: np.mean([x.get("flesch_reading_ease", np.nan) for x in m])),
        avg_latency=("latency_s", "mean"),
    ).reset_index()
    insights["model_summary"] = by_model.to_dict(orient="records")

    # identify prompts with high disagreement (approx via embedding sim)
    # quick heuristic: for each prompt+persona compute pairwise similarity; mark if mean similarity < 0.6
    disagreements = []
    # group by prompt + persona
    grouped = {}
    for r in records:
        k = (r["prompt"], r["persona"])
        grouped.setdefault(k, []).append(r)
    for (prompt, persona), vals in grouped.items():
        texts = [v["text"] for v in vals]
        if len(texts) < 2:
            continue
        emb_model = SentenceTransformer(EMBED_MODEL)
        embs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        sim = np.dot(embs, embs.T)
        # average off-diagonal
        n = sim.shape[0]
        off_diag = sim[np.triu_indices(n, k=1)]
        avg_sim = float(np.mean(off_diag)) if off_diag.size>0 else 1.0
        if avg_sim < 0.6:
            disagreements.append({"prompt": prompt, "persona": persona, "avg_similarity": avg_sim, "n_responses": n})
    insights["disagreements"] = disagreements

    # recommendation heuristic
    recs = []
    for model in by_model.to_dict(orient="records"):
        if model["avg_length"] < 40:
            recs.append(f"{model['model_provider']}: tends to be concise; good for executive summaries.")
        else:
            recs.append(f"{model['model_provider']}: longer outputs; good for tutorials and deep explanations.")
    insights["recommendations"] = recs
    return insights

# -------------------------
# Example runner
# -------------------------
def main_demo():
    # initialize clients (in a real run, supply actual keys)
    openai_client = OpenAIClient()
    hf_client = HFClient()
    anth_client = AnthropicClient()

    orch = Orchestrator(openai_client=openai_client, hf_client=hf_client, anth_client=anth_client)

    prompts = [
        "Summarize the following article into 3 bullet points: 'Machine learning pipelines improve model reproducibility by standardizing preprocessing, training, and evaluation.'",
        "Write a short Python function to remove duplicates from a list while preserving order.",
        "Explain how blockchain helps in tamper-evident logging in 5 lines."
    ]
    models = ["openai", "hf", "anthropic"]
    personas = ["developer", "manager", "critic"]

    print("Running batch... (this will call external APIs — replace keys first)")
    records = orch.run_batch(prompts, models, personas)
    # Save outputs
    with open("multi_ai_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    insights = generate_insights(records)
    with open("multi_ai_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    print("Saved results to multi_ai_results.json and multi_ai_insights.json")
    print("Sample insights:")
    print(json.dumps(insights, indent=2))

if __name__ == "__main__":
    main_demo()
```

---

## 7.4 Explanation of key code parts

* **Client wrappers**: `OpenAIClient`, `HFClient`, and `AnthropicClient` show minimal HTTP calls. In production, use official SDKs and robust error handling.
* **Persona injection**: Personas are added either as `system` messages (OpenAI) or prefixed to the prompt (others).
* **Embeddings and similarity**: Local SentenceTransformer reduces API cost and speeds development. We compute cosine similarities for agreement checks.
* **Metrics**: ROUGE/BLEU/BERTScore require references. If you have gold references (e.g., human summaries), pass them in `reference_texts`. For unreferenced tasks, we rely on semantic similarity and heuristics.
* **Insight heuristics**: Simple rule-based; can be extended with ML.

---

# 8. Example experiments (use cases)

Use cases you can run immediately:

1. **Summarization comparison**

   * Dataset: 50 short news paragraphs.
   * Task: generate 1-sentence summary with personas `manager` and `developer`.
   * Evaluation: ROUGE vs human reference + semantic similarity.
   * Insight: which model produces concise executive summaries vs verbose ones.

2. **Code generation**

   * Dataset: 30 small coding tasks.
   * Task: produce code and unit tests.
   * Evaluation: run unit tests automatically where possible (requires sandbox). Metrics: pass rate, length, complexity.
   * Insight: which model gives more correct code vs more idiomatic code.

3. **QA factuality**

   * Dataset: 40 factual questions.
   * Task: short answer.
   * Evaluation: cross-verify with authoritative sources (not included in code) or use knowledge cutoff heuristics and timestamp checks.

4. **Creative rewriting**

   * Dataset: 20 product descriptions.
   * Task: rewrite for social media.
   * Evaluation: lexical diversity, sentiment uplift, human preference study.

---

# 9. Results interpretation & suggested actionable insights

When you run the orchestrator, expect to see patterns like:

* Models tuned for instruction following (e.g., GPT-style) may produce more structured, stepwise outputs. They often shine in `developer` or `teacher` personas.
* Some models may be more verbose as `critic` persona, offering more caveats — useful for risk assessment.
* Semantic similarity across different providers indicates consensus; low similarity indicates areas requiring human review.
* Actionable recommendations (generated by `generate_insights`) include:

  * Use `model X` for short executive summaries.
  * Use `model Y` for code generation with follow-up unit testing.
  * Flag low-consensus prompts for human review.

---

# 10. Limitations, safety, and reproducibility notes

* **API rate limits & costs**: Calling multiple models quickly may hit rate limits and incur charges. Use small batches when testing.
* **Determinism**: Non-deterministic outputs (temperature, sampling) can cause variance. For reproducible comparisons, set sampling seeds or zero temperature if supported.
* **Factuality**: This system does not automatically verify facts against external sources. For high-stakes outputs, implement an external verification step.
* **Privacy/security**: Don’t send sensitive PII to third-party APIs unless permitted. Use on-prem or privacy-preserving models for sensitive data.
* **Bias**: Models can produce biased outputs. Use adversarial prompts and safety checks.

---

# 11. Conclusion

This document provides a ready-to-run orchestration pipeline for interacting with multiple AI tools, injecting personas, collecting outputs, comparing them using automated metrics, and producing actionable insights. The architecture and code are modular — add more metrics, more clients, or a web dashboard as needed. Use the outputs to guide which model(s) to use for different tasks (summaries, code, QA, creative writing), and to build a human-in-the-loop review process when models disagree.

---

