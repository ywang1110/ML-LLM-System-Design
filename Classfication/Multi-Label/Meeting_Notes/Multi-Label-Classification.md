# Multi-Label Theme Classification for Meeting Notes
## Problem Framing (5-7 min)
### 🎯 1. Clarify the Business Goal

#### 🗂️ TL;DR — The **5 Most Critical Questions** for System Design

1. **What is the trigger timing** (real-time vs batch)?
2. **What’s the scale and latency budget?**
3. **Is the taxonomy stable or evolving?**
4. **How is user feedback captured and used for retraining?**
5. **Which errors matter most (precision vs recall trade-off)?**

---
Here’s a **summary of the most important Problem Framing questions** — the ones that directly influence your **system design decisions** for a *multi-label meeting-notes classifier* (BERT-based).


#### 🧭 1. Business & Objective

* What’s the **main goal of this system**?  
  → e.g., automate tagging, improve search, generate insights, or trigger downstream actions?
* Who are the **end users**, and how will they consume the predictions?
* What defines **success for the business**?  
  → fewer missed “Action Items”? faster document retrieval? reduced manual review time?
* Which **error type is more costly** — missing a theme (false negative) or mis-tagging (false positive)?
* Are there certain “critical” themes that require higher precision or recall (e.g., *Customer Escalation*)?

---

#### 📦 2. Data & Label Taxonomy
* Where do the meeting notes come from?  
  → Zoom transcripts, Notion docs, Confluence pages, internal wikis, etc.?
* What’s the **structure and quality** of these notes?  
  → clean text vs noisy transcriptions; presence of sections or bullet points?
* How much **labeled data** do we have today?  
  → Do we need weak supervision, heuristics, or active learning to bootstrap?
* How many labels exist, and how are they defined?  
  → Are they mutually exclusive, hierarchical, or overlapping?
* **Is the taxonomy fixed, or will new labels be added or changed over time?** ✅

  * If yes: **how frequently** do new themes appear?
  * Who defines or approves new labels (business vs ML team)?
  * Do we need backward compatibility between old and new label versions?
  * Should the system support **incremental retraining** or **zero-shot expansion** for new labels?
* Are there any compliance or PII concerns (names, emails, client data) that require anonymization?

---

#### ⚙️ 3. Operational Constraints

* When is classification triggered   
— right after the meeting ends, upon saving the note, or batch overnight?
* What’s the expected **volume and throughput** (e.g., number of notes/day)?
* What’s the **latency** requirement (e.g., <100 ms per note, or batch acceptable)?
* Does the model need **interpretability** (evidence sentences, highlight rationales)?
* Is there a **human-in-the-loop review** process for low-confidence predictions?
* What **infrastructure** will **serve** this model (CPU vs GPU, on-prem vs cloud)?

---

#### 🔁 4. Feedback & Retraining Loop

* Will users be able to **correct labels**?
  → If so, how is feedback captured (UI? logs?)
* How often do we plan to **retrain the model**?
  → On schedule (weekly/monthly) or triggered by drift?
* Do we need to detect **data or vocabulary drift** (new terms, project names, abbreviations)?
* Should retraining handle **label schema changes** automatically (taxonomy versioning)?

---

#### 🧱 5. Integration & Downstream

* Where will predictions be stored or consumed?  
  → task system, search index, analytics dashboards, etc.
* Are there existing APIs or schema contracts that depend on label names/IDs?
* Do downstream systems need stable label IDs (requiring label-version mapping)?
* Should we provide confidence scores or only discrete labels?

### 🎯 2. Establish a Business Objective
Organize and operationalize meeting knowledge by **automatically tagging notes with key business themes** (Action Items, Risks, Escalations, etc.),
so teams can retrieve information faster, automate follow-ups, and reduce manual effort.

Success =  
↓ Missed action items or escalations  
↓ Manual review cost  
↑ Task automation rate  
↑ Search and discovery satisfaction

### 🎯 3. Define a ML Objective
Minimize binary cross-entropy over multi-label themes.

## High Level Design (2-3 min)
| **Stage**                             | **Meeting Notes Theme Classification System**                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **1. Ingestion**                      | Collect meeting notes from tools like Zoom, Notion, or Confluence, along with metadata (organizer, date, team, project).       |
| **2. Preprocessing**                  | Clean text, remove boilerplate, **redact PII**, chunk long notes, and detect sections (Agenda, Action Items, Decisions).           |
| **3. Label Generation**               | Combine a small human-labeled “gold set” with weak signals such as section titles, task links, and thematic keywords.          |
| **4. Feature & Representation Layer** | Use BERT/DistilBERT embeddings, plus section-type and metadata embeddings (meeting type, time, organizer).                     |
| **5. Model Training**                 | Multi-label classifier trained with BCE loss, per-label thresholds, and calibration to balance precision vs recall.            |
| **6. Serving & Prediction**           | Expose an API that receives note text, runs preprocessing and inference, then outputs labels, confidences, and evidence spans. |
| **7. Feedback & Retraining**          | Incorporate user corrections and new labeled data into retraining; support taxonomy versioning for new or evolving themes.     |
| **8. Monitoring & Drift**             | Track label distribution, confidence drift, and vocabulary drift; trigger retraining or taxonomy updates when needed.          |


## Data and Features (10 min)
### 🧩 Data
1. **Primary content**:   
meeting notes and transcripts from Zoom, Notion, or Confluence.
2. **Metadata**:   
meeting title, organizer, attendees, date, team/project, source app.
3. **Labels:**  
    * a small manually labeled **“gold set”** 
    * weak signals (section titles, task links, keywords like budget, risk).
4. **Feedback:**  
user corrections or downstream actions (e.g., tasks auto-created).
5. **Privacy:**  
redact emails, phone numbers, ticket IDs before storage or training.

### ✳️ Features
1. **Textual features:**  
    * BERT/DistilBERT embeddings over note chunks (384 tokens, stride 128).
    * Section type indicators (Agenda, Decisions, Action Items).
    * Keyword and pattern matches (e.g., due: dates, @mentions).
2. **Structural features:**   
bullet lists, tables, or checkboxes → strong signal for Action Items.
3. **Metadata embeddings:**   
meeting type, organizer, project, or date; help model domain context.
4. **Cross-document context:**   
tags from previous meetings in the same series, or linked Jira/PRs.
5. **Calibration & uncertainty:**  
per-label thresholds, temperature scaling, confidence scores for human review.

## Modelling
### Benchmark Models
#### **Traditional ML Baseline — TF-IDF + One-vs-Rest Logistic Regression / Linear SVM**

* **Idea:** Represent each note as a sparse TF-IDF vector, then **train one binary classifier per theme label** (One-vs-Rest).
* **Process:**

  1. Convert text → TF-IDF features (word importance).
  2. Train Logistic Regression or Linear SVM for each label.
  3. Combine outputs → multi-label predictions.
* **Why it’s useful:**

  * Fast, simple, interpretable.
  * Works with small labeled data.
  * Provides a strong baseline to measure BERT improvements.
* **Limitations:**

  * Ignores context and word order.
  * Poor for long, noisy, or evolving language.

**Use it early** to validate data quality, feature coverage, and expected gains from transformer models.

### Model Selection
| Model                                     | Pros                                         | Cons / When to Avoid                        |
| ----------------------------------------- | -------------------------------------------- | ------------------------------------------- |
| **TF-IDF + LogReg/SVM**                   | Fast, interpretable, good sanity baseline    | No context, weaker recall on long notes     |
| **DistilBERT**                            | Good accuracy–**latency** trade-off (≈80 ms p95) | Slight recall loss vs full BERT             |
| **BERT-base**                             | Stronger recall and semantics                | Slower (≈2× DistilBERT latency)             |
| **Longformer / ModernBERT**               | Handles **>512 tokens**                          | Heavier compute, use if most notes are long |
| **Domain-adapted BERT**                   | Better jargon understanding                  | Needs extra data for pre-training           |
| **LLM zero-shot (e.g., GPT-4, Gemma-3N)** | Useful for new labels or evaluation          | High cost, unpredictable latency            |

#### Decision Logic (simple rule of thumb)
1. Cold-start / few labels → TF-IDF or weak rules.
2. Production MVP → DistilBERT (balanced).
3. Enterprise / longer notes → Longformer or Domain-adapted BERT.
4. New or unseen labels → use zero-shot LLM fallback, then retrain.

### Model Architechture
#### Components
| Component           | Function                                          | Notes                                   |
| ------------------- | ------------------------------------------------- | --------------------------------------- |
| **Tokenizer**       | WordPiece/BPE; truncation & overlap for long docs | Keeps context continuity                |
| **Encoder**         | DistilBERT / BERT-base                            | Captures contextual semantics           |
| **Pooling Layer**   | Aggregate multiple chunks                         | `max`, `mean`, or **attention pooling** |
| **Classifier Head** | Linear + Sigmoid (K outputs)                      | Allows multi-label (not softmax)        |
| **Loss Function**   | `BCEWithLogitsLoss` or Focal Loss                 | Handles imbalance per label             |
| **Calibration**     | Temperature / Isotonic scaling                    | Ensures probabilistic reliability       |
| **Threshold Layer** | Per-label τₖ tuned for F1 / business precision    | Critical for deployment consistency     |

`note` Because **DistilBERT** can only **handle 512 tokens per input**, we split long meeting notes into overlapping chunks. Each chunk is **encoded separately**, and we use a **pooling layer** — typically max, mean, or attention pooling —to **aggregate them into a single document-level embedding** before passing it to the classifier head.

`note` Traditional encoder models like BERT or DistilBERT have a 512-token limit,
which forces us to chunk long documents and pool embeddings.
But modern LLMs like GPT-4o or Claude 3.5 support 128K–**200K tokens**,
and Gemini 1.5 even scales to 1 million, making long-context reasoning and summarization possible without chunking.

### Loss Function

#### 1️⃣ Problem Type

Each meeting note can belong to **multiple themes** simultaneously
(e.g., *Action Items*, *Budget*, *Hiring*, *Risks*).
So it’s **multi-label**, not multi-class.  

👉 That means **each label is an independent binary decision**.

---

#### 2️⃣ Formula

`BCEWithLogitsLoss` combines **sigmoid activation** and **binary cross-entropy loss** into a single, **numerically stable operation**:

$\mathcal{L} = -\frac{1}{K}\sum_{k=1}^{K}[ y_k \cdot \log(\sigma(x_k)) + (1 - y_k)\cdot\log(1 - \sigma(x_k))]$

where

* $x_k$: raw model logit for label **k** (before sigmoid)
* $\sigma(x_k) = \frac{1}{1 + e^{-x_k}}$: sigmoid function
* $y_k \in {0,1}$: ground truth label
* $K$: number of labels

👉 **You do not manually apply sigmoid** in code — `BCEWithLogitsLoss` handles it internally.

---

#### 3️⃣ Weighted version (for class imbalance)

PyTorch also supports per-label weights via `pos_weight`:


$\mathcal{L} = -\frac{1}{K}\sum_{k=1}^{K}[w_k^+ , y_k \cdot \log(\sigma(x_k)) + (1 - y_k)\cdot\log(1 - \sigma(x_k))]$

where $w_k^+ = \frac{N_{neg}}{N_{pos}}$.

---

#### 4️⃣ Implementation

```python
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
logits = model(inputs)           # [batch_size, K]
loss = criterion(logits, targets.float())
```

---

#### 5️⃣ Numerical stability form (internal computation)

Internally, PyTorch computes it as:

$\text{loss}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$

This avoids overflow or underflow when `x` is very large or small —
one reason we prefer the *WithLogits* version.

---

#### 💬 Interview Sound Bite

> “We use `BCEWithLogitsLoss` because it fuses sigmoid and BCE in a single, numerically stable step.
> It treats each label independently, supports per-label weighting for imbalance,
> and is the standard loss for multi-label classification tasks like ours.”

## Inference

#### ⚙️ Inference Pipeline

---

##### **1️⃣ Input**

* Raw meeting note text (often long) + metadata
  → e.g. title, organizer, meeting time, team, project.

---

##### **2️⃣ Preprocessing**

* **PII redaction**: mask emails, phone numbers, IDs.
* **Chunking:** split into overlapping windows (e.g. 384 tokens, stride 128) because DistilBERT supports up to **512 tokens**.
* **Section detection:** mark “Agenda”, “Action Items”, “Decisions” as contextual features.

---

##### **3️⃣ Encoding**

Each chunk → tokenizer → **DistilBERT encoder** (shared weights).
Output: one embedding or logits vector per chunk.

---

##### **4️⃣ Pooling Across Chunks**

Aggregate chunk representations into a single document-level vector:

* **Max pooling** (default): capture the strongest signal per label.
* **Mean pooling**: smoother, context-averaged view.
* **Attention pooling** (optional): learnable weights **highlighting important chunks** (e.g., “Action Items”).

---

##### **5️⃣ Classification**

Document embedding → linear classifier → K logits
(one per theme label).

Apply **sigmoid activation** internally (if using `BCEWithLogitsLoss` trained model).  
Output: probabilities ($\hat{p}_k \in [0,1]$) for each label (k).

---

##### **6️⃣ Calibration & Thresholding**

* **Temperature / isotonic scaling** per label → better probability accuracy.
* **Per-label thresholds ( $\tau_k$ )** tuned on validation set
  → optimize F1 or meet precision/recall targets.

  * *Example:* `τ_ActionItems = 0.4`, `τ_Budget = 0.6`.

---

##### **7️⃣ Post-processing**

* **Apply thresholds:**

$\text{predict label } k \text{ if } \hat{p}_k \ge \tau_k$

* **Add business rules:**

  * e.g., if `Escalation` is predicted with low confidence → send to human review.
  * if `Action Items` detected → auto-create task in downstream system.
* **Evidence extraction:**
  Highlight the chunk sentences contributing highest attention or logits.

---

##### **8️⃣ Output Schema**

Example response from API:

```json
{
  "note_id": "1234",
  "predictions": [
    {"label": "ActionItems", "score": 0.93, "evidence": "Follow up with Finance"},
    {"label": "Budget", "score": 0.71}
  ],
  "taxonomy_version": "v2",
  "model_version": "2025-10-05_1530"
}
```

---

##### **9️⃣ Serving & Optimization**

* **Batch small requests** for GPU/ONNX efficiency.
* **Cache** repeated documents by hash.
* **ONNX / BetterTransformer** → reduce latency to **~80 ms p95.**
* **Microservice endpoint** (HTTP/gRPC) with request/response logging for traceability.

---

### 💬 **Interview Sound Bite**

> “At inference time, we preprocess and chunk the note, run each chunk through DistilBERT, pool logits to document level, apply calibrated sigmoid scores, and threshold per label.
> The pipeline outputs labels with confidences and evidence spans, all within an 80 ms latency budget.”

## Evaluation
### **1️⃣ Split Strategy**

* **Time-based split** → train on past, test on future.

  * Avoids data leakage.
  * Simulates real deployment drift.

---

### **2️⃣ Core Offline Metrics**

| Metric                           | Purpose                                                  |
| -------------------------------- | -------------------------------------------------------- |
| **Macro / Micro F1**             | Main quality indicator for multi-label tasks.            |
| **Per-label Precision / Recall** | Track critical themes (*Action Items*, *Escalation*).    |
| **PR-AUC / mAP**                 | Handle class imbalance.                                  |
| **Calibration (ECE)**            | Check probability reliability after temperature scaling. |

✅ Target example:
Macro-F1 ≥ 0.80, Recall(*Action Items*) ≥ 0.9, Precision ≥ 0.8

> “Micro-F1 measures overall accuracy weighted by sample count — good for global performance.
Macro-F1 averages F1 per label equally — good for checking long-tail or rare labels.
In multi-label systems like ours, we monitor both: Micro-F1 for stability, and Macro-F1 to ensure rare themes aren’t ignored.”

---

### **3️⃣ Threshold & Calibration**

* Tune **per-label thresholds** τₖ for best F1 or business precision floor.
* Apply **temperature / isotonic scaling** for probability calibration.

---

### **4️⃣ Error Analysis**

* Review **Top FP/FN** cases per label.
* Analyze **slice performance** (source app, team, length).
* Detect long-tail labels & drift.

---

### **5️⃣ Online & Monitoring**

| Aspect               | Metric / Goal                                              |
| -------------------- | ---------------------------------------------------------- |
| **Shadow / Canary**  | Macro-F1 drop < 1%, per-label guardrails hold.             |
| **Business KPIs**    | ↓ Missed Action Items, ↓ Review Time, ↑ Auto-task success. |
| **Drift Monitoring** | Label prevalence, confidence, and vocab drift alarms.      |

> “Online, we first run shadow and canary tests to ensure no major regression — macro-F1 drop under 1% and guardrails hold for critical labels.
Once live, we track business KPIs like fewer missed action items and shorter review time,
and continuously monitor drift in label frequency, confidence, and vocabulary to trigger retraining when needed.”
---

### 💬 **Interview Sound Bite**

> “We use time-based splits to mirror production drift,
> optimize per-label thresholds and calibration,
> track macro/micro-F1 and per-label recall,
> and monitor business KPIs like missed Action Items and reviewer effort post-launch.”

