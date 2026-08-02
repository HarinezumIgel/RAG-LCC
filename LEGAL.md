# Legal and Compliance Notes

RAG‑LCC is designed for experimentation and learning, not for production deployment.
Use it with care, especially when enabling features that access external services.

This document is provided for informational purposes only. It does not constitute legal
advice, compliance advice, or professional guidance of any kind. The authors and
maintainers of RAG‑LCC are not lawyers and do not provide legal services.

RAG‑LCC is an experimental research framework intended solely for laboratory evaluation
and experimentation. It is **not intended for production use**, regulated processing,
or deployment in safety‑critical or compliance‑sensitive environments.

Operators must obtain independent legal advice before using this software in any context
where legal, regulatory, contractual, or statutory obligations may apply.

---

## 📖 Definition — “Compliance” (RAG‑LCC)

In the context of RAG‑LCC, **“compliance”** refers solely to a configurable, algorithmic
*detection and scoring pipeline* used to flag, score, redact, or optionally block content
based on operator‑supplied rules (e.g. keyword lists, thresholds, and consensus criteria).

This term is used **only in a technical sense**.

RAG‑LCC **does not**:

- determine legal compliance,
- guarantee regulatory conformity,
- provide legal classification,
- certify data suitability, or
- replace human judgment.

All detection results are probabilistic. False positives and false negatives will occur.
Final decisions regarding enforcement, disclosure, remediation, or use of outputs must be
made by a qualified human operator.

---

## 📋 Key Principles

- **Detection only**
  RAG‑LCC performs automated detection and scoring. It does not make legal,
  regulatory, or contractual determinations.

- **Operator control**
  All rules, thresholds, and configurations are defined and controlled by the operator.

- **Human responsibility**
  Any blocking, redaction, reporting, or external disclosure must be reviewed and approved
  by a human. RAG‑LCC does not autonomously enforce outcomes.

- **Experimental limitations**
  The software may contain errors, omissions, or design limitations and has not been
  validated for production use.

---

## 🔒 Data Protection and Privacy

Operators are solely responsible for determining:

- whether personal data is processed,
- whether a lawful basis exists,
- whether a DPIA or other assessment is required,
- how retention, access control, and deletion are handled.

The default configuration is local‑only. Any use of external model endpoints, telemetry,
or network access is explicitly configured by the operator and is outside the control of
the authors.

---

## 🌐 Web / Internet Search — Privacy Warning

RAG‑LCC includes an **optional internet-search feature** which is
**disabled by default**. When an operator enables it, RAG‑LCC sends the
user's query to an external third-party search provider (currently
DuckDuckGo). For the exact configuration keys and activation paths see
[INSTALL.md § Enable Internet (Web) Search](INSTALL.md#-enable-internet-web-search-optional)
and [SECURITY.md](SECURITY.md#-web--internet-search); this section describes only the
privacy / liability implications.

When the feature is active:

- the operator's **IP address, query text, and timing metadata** are
  disclosed to the search provider and are subject to that provider's
  own privacy policy and retention practices, entirely outside the
  control of the authors;
- queries may contain **sensitive, proprietary, or personal data** typed
  by the user — operators must ensure such disclosure is lawful (e.g.
  GDPR Art. 13/14 transparency obligations, or equivalent);
- if the optional "fetch pages" mode is enabled, the operators of the
  fetched websites additionally receive the IP address and standard
  HTTP request headers;
- the authors make **no representation or warranty** regarding the
  transport (TLS / cleartext), confidentiality, or data-handling
  practices of any third-party search provider or HTTP library used to
  reach it — operators must independently verify these properties for
  the installed version of any dependency before relying on them.

The operator is **solely responsible** for all data flows resulting from
enabling the feature, including assessing whether a DPIA or equivalent
is required in the deployment context.

---

## 🚫 No OpenAI / OpenWebUI Feature Parity

`RAGChatService` exposes only a deliberately narrow subset of the OpenAI
REST surface (`POST /v1/chat/completions`, `GET /v1/models`, Bearer
authentication). Tool / function calling, vision input, audio in/out,
embeddings, image generation, file uploads, structured-output / JSON
mode, and the OpenWebUI Knowledge passthrough are **not** implemented.
This restriction is intentional and is part of the compliance trust
boundary — operators must not rely on behavioural parity with hosted
OpenAI, Ollama, or stock OpenWebUI back-ends. Engineering rationale and
the full list of intentional restrictions vs. known gaps is documented
in [ARCHITECTURE.md](ARCHITECTURE.md#-ragchatservice--openai-surface-intentional-restrictions-vs-known-gaps).

---

## 📜 Model Licenses and Permissions

- RAG‑LCC does **not** bundle or redistribute LLMs, embedding models, or cross‑encoders.
- Operators must independently obtain, review, and accept all applicable model licenses
  via the model owner's official distribution channel (e.g. Hugging Face, Ollama, OpenWebUI).
- Export controls, sanctions regimes, and usage restrictions remain the sole
  responsibility of the operator.

RAG‑LCC makes no representation that any referenced model is legally usable for a given
purpose or jurisdiction.

---

## 📦 Third‑Party Dependencies

RAG‑LCC does **not** bundle, ship, or redistribute any third‑party Python packages,
pre‑trained models, translation packages, or external tools.

All dependencies are obtained independently by the operator from upstream sources
(PyPI, GitHub, Hugging Face, Open WebUI, vendor websites, etc.).

A list of dependencies and their licenses may be provided in `3rdPartyLicenses/` for
**informational and attribution purposes only**. This list may be incomplete, outdated,
or inaccurate and is not legally authoritative.

By installing dependencies, users enter into a direct licensing relationship with each
dependency’s author(s) and are solely responsible for:

- reviewing and accepting license terms,
- ensuring license compatibility,
- complying with copyleft or attribution obligations,
- verifying export‑control and sanctions constraints.

The authors accept no liability arising from the installation or use of third‑party
software.

---

## 🚫 No Support, No SLA, No Maintenance Obligation

## 🚫 No Support and No Maintenance Commitment

RAG‑LCC is provided without any obligation to provide support, respond to
inquiries, fix defects, address vulnerabilities, maintain compatibility,
or deliver updates.

There is no **service-level agreement (SLA)**, **maintenance commitment**, or
**security support** of any kind.

---

## ⚖️ Warranty Disclaimer and Limitation of Liability

This software is provided **“as‑is”** and **“as available”**, without warranty of any
kind, whether express, implied, or statutory, including but not limited to warranties of
merchantability, fitness for a particular purpose, non‑infringement, or accuracy.

To the fullest extent permitted by applicable law, the authors and maintainers disclaim
all liability for any direct, indirect, incidental, consequential, or regulatory damages
arising from the use of this software.

All risk arising from use, configuration, deployment, or interpretation of outputs rests
entirely with the operator.

---

## 🏛️ Governing Law and Jurisdiction

To the extent permitted by applicable law, any dispute arising out of or in connection
with this software shall be governed by Swiss law, excluding its conflict‑of‑laws rules,
and subject to the exclusive jurisdiction of the courts of Zurich, Switzerland.

---

## ⚠️ Final Responsibility Notice

Compliance with all applicable laws and regulations — including emerging or future
AI‑specific regulatory regimes — rests entirely with the user.

RAG‑LCC is a research tool. It does not confer compliance, certification, or legal safety.

### Analytical Scope and Non‑Delegation of Responsibility

RAG‑LCC provides analytical and experimental tooling only.

All outputs, classifications, scores, alerts, or signals produced by this
software are heuristic and probabilistic in nature. They do not constitute
decisions, recommendations, instructions, or authoritative judgments.

RAG‑LCC does not perform actions, enforce policies, or make determinations on
behalf of users. Responsibility for interpreting outputs and taking any
subsequent actions rests solely with the user or operator.

The author does not control downstream usage of this software and assumes no
liability for outcomes resulting from modification, integration, automation,
or application of its outputs in operational, legal, compliance, or decision‑
making contexts.

---
