# Security Considerations

**Disclaimer.** This document is provided for informational purposes only and may contain
errors, omissions, or outdated information. It does not constitute security advice,
professional advice, or legal advice.

RAG‑LCC is experimental research software provided “as‑is” without warranty of any kind.
See [LICENSE](LICENSE) and [LEGAL.md](LEGAL.md) for full terms and disclaimers.

---

## 🎯 Scope and Intent

RAG‑LCC is an **experimental, local‑only lab framework** intended for evaluation and
experimentation. It has **not undergone a formal security audit** and must not be relied
upon for production use, regulated processing, or security‑critical workloads.

All security decisions, risk assessments, and mitigations remain the responsibility of
the operator.

---

## 🏗️ Architecture (Security‑Relevant Characteristics)

| Aspect | Default Posture |
| ------ | ----------------- |
| Network exposure | RAG‑LCC's console applications do not open listening ports. The optional `RAGChatService.py` **does** open a listening socket (default `127.0.0.1:11435`) via Uvicorn to serve an OpenAI‑compatible REST API. If run in Docker, the configured port (default `11435`) is forwarded. The operator controls bind address and port through configuration and is responsible for any network‑security implications. Note: on startup, Python or its dependencies may attempt a brief connection to `[::1]` (IPv6 loopback) to probe whether IPv6 is available on the host. This is a local‑only probe. |
| External calls | **None by default.** When `web_search` is enabled (`WEB_SEARCH_MODE = "1"` in `Config_Internet_Env.py`), user queries are transmitted to an external search provider (DuckDuckGo). See [🌐 Web / Internet Search](#-web--internet-search) below. |
| Data storage | Local filesystem and local ChromaDB only. When `mark_text` is enabled, highlighted document bytes are held in process memory and (CLI) written to a short-lived OS temp directory that is deleted at process exit. No persistent copies are created by the framework. |
| Highlighted-doc HTTP tokens (`RAGChatService`) | When `_MARKED_DOCS["enabled"]` is `True`, highlighted document bytes are served via `GET /marked/<token>.<ext>`. Tokens are 256-bit `secrets.token_urlsafe` values. The route is outside `/v1/*` so the Bearer middleware is intentionally bypassed — the token IS the access credential. Entries auto-expire (default 1800 s). Operators who require stricter isolation should place the service behind an authenticated reverse proxy. |
| Telemetry | None |
| Authentication | Not applicable (single‑user, local execution) |

Actual behavior may vary depending on configuration, environment, and third‑party
components.

---

## 📦 Dependency and Supply‑Chain Responsibility

RAG‑LCC does **not** bundle, ship, or redistribute:

- Python packages,
- LLMs or ML models,
- translation packages,
- OCR engines,
- external tools,
- GUI front‑ends (e.g. OpenWebUI).

All dependencies are installed independently by the operator from upstream sources.
The authors have no control over their code, security posture, or supply chain.

Operators are solely responsible for:

- vetting dependencies,
- monitoring vulnerabilities (CVEs),
- applying updates or mitigations,
- evaluating trustworthiness of maintainers,
- running security or vulnerability scanners.

The authors make no representation regarding the security of any third‑party component
and accept no liability for vulnerabilities or compromises arising from their use.

---

## 🗃️ Data Handling Considerations

- Documents remain at their original retrieval location and are processed locally.
- Content is stored in a local vector database as configured by the operator.
- Detection and filtering mechanisms are probabilistic and may fail.- When `mark_text` is enabled, highlighted copies of retrieved source documents are held in process memory. In CLI mode (`RAGChat`), a copy is written to the OS temp directory only when the user explicitly opens a file; the temp directory is deleted at process exit. In service mode (`RAGChatService`), bytes are held in the `MarkedDocsStore` in-memory cache and served over HTTP for the configured TTL before expiry; they are never written to disk.
RAG‑LCC’s filtering and detection features are **diagnostic aids**, not security controls.

---

## 🌐 Web / Internet Search

This feature may transmit query data and metadata to external services.
See [LEGAL.md — 🌐 Web / Internet Search: Privacy Warning](LEGAL.md#-web--internet-search--privacy-warning) for full details.

**Mitigations the operator should consider:**

- Keep `WEB_SEARCH_MODE = "0"` (default) unless internet access is appropriate.
- Route outbound traffic via a corporate proxy to maintain visibility and control.
- Notify users before enabling this feature in any shared environment.
- Review the search provider's privacy policy and data-retention practices.

This feature is **disabled by default**. See also [LEGAL.md — 🌐 Web / Internet Search: Privacy Warning](LEGAL.md#-web--internet-search--privacy-warning).

---

## ⚠️ Known Limitations

- No formal security audit has been performed.
- The codebase may contain defects or vulnerabilities.
- Detection does not guarantee prevention.
- No guarantees are made regarding correctness, completeness, or timeliness of results.

---

## 🔍 Network Activity Observation (Optional)

RAG‑LCC includes an optional Python‑level socket activity tracer (`NetworkTracer`) that
can log certain DNS and connection attempts when explicitly enabled by the operator.

This mechanism **may assist** operators in observing Python‑level socket usage but:

- does not capture activity from native extensions or subprocesses,
- does not guarantee full visibility,
- is not a security control.

It is a diagnostic tool only.

---

## 📨 Vulnerability Reporting

Vulnerability reports may be submitted via the project’s issue tracker.

The authors may, but are not obligated to:

- review reports,
- respond,
- provide remediation,
- disclose timelines.

No guarantee of response, fix, or acknowledgment is provided.

---

## 🚫 Out of Scope

The following are explicitly outside the scope of this project:

- operating‑system security,
- network perimeter security,
- encryption at rest or in transit,
- access control and authentication,
- backup and disaster recovery,
- regulatory compliance validation.

---

## 🚫 No Support and No Security Commitments

RAG‑LCC is provided without any obligation to:

- provide security support,
- respond to security inquiries,
- patch vulnerabilities,
- maintain secure configurations,
- provide updates or advisories.

There is **no security SLA**, **no support agreement**, and **no maintenance commitment**.

---

## ⚖️ Limitation of Liability

To the fullest extent permitted by applicable law, the authors and maintainers disclaim
all liability for any security incidents, data loss, or damages arising from the use or
misuse of this software.

All risk is assumed by the operator.
