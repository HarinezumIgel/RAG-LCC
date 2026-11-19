# Security Considerations

**Disclaimer.** This document is provided for informational purposes only and may contain
errors, omissions, or outdated information. It does not constitute security advice,
professional advice, or legal advice.

RAG‑LCC is experimental research software provided “as‑is” without warranty of any kind.
See [LICENSE](LICENSE) and [LEGAL.md](LEGAL.md#-warranty-disclaimer-and-limitation-of-liability) for full terms and disclaimers.

---

## 🎯 Scope and Intent

RAG‑LCC is an **experimental lab framework** intended for evaluation and
experimentation. It has **not undergone a formal security audit** and must not be relied
upon for production use, regulated processing, or security‑critical workloads.

RAG‑LCC is intended to run by a **single user**. See [Thread Safety in
ARCHITECTURE.md](ARCHITECTURE.md#-thread-safety).

All security decisions, risk assessments, and mitigations remain the responsibility of
the operator.

---

## 🏗️ Architecture (Security‑Relevant Characteristics)

**RAG-LCC is not guaranteed to be client-only or free of listening sockets.**
Running RAGChat or RAGChatService can bind a port and accept incoming HTTP
requests as described below. Local models, loopback binding and outbound
offline/download settings do not mean that no server is listening.

| Aspect | Default Posture |
| ------ | ----------------- |
| Network exposure | The optional RAGChat CLI HTML listener enforces `127.0.0.1`, using configurable port `11436` by default (`_LOCAL_SERVE_DOCS.port`). RAGChatService shares its Uvicorn listener between API and document routes on port `11435` by default. Its configured bind defaults to `127.0.0.1`, but is not restricted to loopback by runtime validation. Proxies and forwarding can expose even a loopback listener remotely. See [In-memory document servers](#in-memory-document-servers-risks-and-safe-operation). Python/dependencies may also probe IPv6 loopback locally. |
| External calls | Deployment disables the web-search master switch, but development settings may differ. Hugging Face access depends on configuration, and model endpoints can be remote. Queries sent to the web leg go to DuckDuckGo. See [🌐 Web / Internet Search](#-web--internet-search) below. |
| Data storage | Indexes and logs can persist locally. With `mark_text`, highlighted document bytes are held in memory; the CLI may also write copies to a temporary directory, with cleanup attempted on normal interpreter shutdown. |
| Highlighted-doc HTTP tokens | Service and CLI document links require both a 256-bit document token and a 192-bit per-startup session capability: `GET /marked/<document-token>.<ext>?token=<session-token>`. They are generated with `secrets.token_urlsafe(32)` and `secrets.token_urlsafe(24)`. The route is outside `/v1/*`; both capabilities replace browser login, not user isolation. Missing/invalid capabilities return 404 before fetching or consuming a document. Anyone holding a full link can use it until expiry. Built-in document access logs redact credentials, and responses disable caching and referrer transmission; reverse proxies must also redact document paths and queries. Service runtime controls live in `_SERVE_DOCS` (`ttl_seconds`, `single_use`, `max_total_mb`, `cors_origins`, `public_base_url`). |
| Remote data export | Web search, remote model endpoints, and other configured network services may transmit data externally. |
| Authentication | Optional Bearer authentication covers `/v1/` service routes; marked-document links use their own access tokens. |

Actual behavior may vary depending on configuration, environment, and third‑party
components.

---

## In-memory document servers: risks and safe operation

**Leave both document-serving switches disabled unless the feature is needed.**
Starting either server makes cached document content accessible over HTTP to
clients that can reach the listener and possess a complete capability URL.
"In memory" describes the application's cache, not a confidentiality guarantee.

| Switch in `Config_Internet_Env.py` | What enabling `"1"` exposes |
| --- | --- |
| `SERVE_RAGCHAT_GROUNDED_DOCS` | Running RAGChat starts a separate loopback-only HTTP listener that accepts document requests, normally at `127.0.0.1:11436`. It serves HTML sources, grounded answers and unverified answer previews. `_LOCAL_SERVE_DOCS` controls its port, TTL and cache size. Non-HTML CLI documents retain their existing file links/picker. |
| `SERVE_RAGCHATSERVICE_GROUNDED_DOCS` | Running RAGChatService accepts document requests for highlighted sources and grounded-answer HTML through `/marked` on its existing API listener, normally port `11435`; no additional service port is opened. `_SERVE_DOCS` controls its cache and link settings. Service startup also requires `SERVE_OPENWEBUI_CHAT="1"`. |

Setup defaults both document switches to `"0"` and deployment forces them off.
The checked-in development configuration currently sets both to `"1"` and also
sets `SERVE_OPENWEBUI_CHAT="1"`; inspect the
actual assignments rather than assuming a development checkout is disabled.
Change the flags in `Config_Internet_Env.py` and restart the affected application.
Disabling one document server does not disable the other or stop the chat API.
In particular, `SERVE_RAGCHATSERVICE_GROUNDED_DOCS="0"` disables document
delivery, not RAGChatService's API listener. To prevent the service from starting,
set `SERVE_OPENWEBUI_CHAT="0"` as well. These flags take effect when the
corresponding application starts; editing them does not stop a running process.

### Local binding does not guarantee local access

- **RAGChat CLI enforces loopback binding. RAGChatService currently only defaults
  to it.** Its `_MODELS.ragchatservice._RAGCHATSERVICE.HOST` is configurable; neither
  the current Setup prompt nor the service runner rejects non-loopback values.
  Keep the service on `127.0.0.1`/`localhost` unless exposure is deliberately
  managed. Do not treat the default as an enforced local-only restriction.
- A reverse proxy, SSH tunnel, VS Code forwarded port, or other relay can make
  a loopback-bound listener reachable from another machine. If a remotely
  accessible proxy forwards `/marked`, it exposes document delivery as well as
  chat. OpenWebUI login and the API's `/v1/` Bearer authentication do **not**
  automatically protect these document routes.
- Inside Docker, loopback belongs to the container; in a browser it belongs to
  the browser's machine. Dev Containers forward `11435` and `11436`; keep the
  host-side listeners local and any shared forwarding private. Standard Docker
  published ports need a non-loopback container bind. If using that mode, limit
  the published host address, for example `-p 127.0.0.1:11435:11435`; this does
  not isolate other containers on the same network.
- `_SERVE_DOCS.public_base_url` only changes the URL printed in answers. It does
  not change the bind, create a tunnel, enable HTTPS, or add authentication.
  For remote browsers, use an explicitly controlled tunnel or HTTPS reverse
  proxy, and apply access restrictions to `/marked` as well as `/v1/`.

### Capability links are credentials, not user isolation

- Each URL carries a random document token and a per-startup session capability:
  `/marked/<document-token>.<ext>?token=<session-token>`. Both are required, but
  **anyone holding the complete URL can fetch the document** while it remains
  available. There is no per-user or per-chat ownership check. The service's
  session capability is shared across its store lifetime, not a user login.
- Links can leak through chat histories, shared conversations, screenshots,
  copied URLs, browser history, terminal transcripts, monitoring and proxy logs.
  Built-in document access logging suppresses/redacts credentials, but that
  does not remove links from chat output or other logging systems. Redact the
  entire document path and query in reverse-proxy/access-monitoring logs.
- Built-in document delivery uses plain HTTP. If relayed over an untrusted
  network without encryption, both credentials and document content can be
  intercepted. Use HTTPS at the remote entry point and protect the path from
  the proxy/tunnel to the application too.
- CORS settings are browser policy, not authentication or a network allowlist;
  they do not stop direct HTTP clients. `Referrer-Policy: no-referrer` and
  `Cache-Control: private, no-store` reduce accidental leakage but cannot prevent
  an authorized recipient from saving, copying or redistributing content.

### Sensitive content, retention and resource limits

- Served content can include full highlighted source documents, not just the
  passages shown in an answer. Verify that every intended recipient may access
  the entire document. Grounding indicates evidence matching, not permission
  to disclose data or a security review of document contents.
- Use a short `ttl_seconds`, a conservative `max_total_mb`, and, where compatible
  with browser/PDF viewers, `single_use=True`. Expiry is measured from creation,
  not the last click. Single-use links may be consumed by previews or scanners
  before the intended reader; neither expiry nor single use revokes a saved copy.
- Restarting invalidates old capability links. It does not erase downloaded
  documents, existing logs, chat histories or screenshots. An in-memory cache
  also does not prevent OS paging or crash dumps, or guarantee secure erasure.
- Cache limits bound stored payload bytes, not total process memory, active HTTP
  connections or document-processing work. Even local processes can reach a
  loopback listener. These experimental servers are not hardened public file
  hosting; keep them off the public Internet and apply proxy authentication,
  request/rate limits and firewall controls if remote access is necessary.

For configuration and forwarding details, see
[CLI delivery](CONFIGURATION_REFERENCE.md#cli-delivery) and
[Service delivery](CONFIGURATION_REFERENCE.md#service-delivery-ragchatservice).

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

- Documents remain at their original retrieval location; configured remote services may receive content or queries.
- Content is stored in local indexes as configured by the operator; logs may also contain sensitive information.
- Detection and filtering mechanisms are probabilistic and may fail.
- When `mark_text` is enabled, highlighted copies of retrieved source documents are held in process memory. In CLI mode (`RAGChat`), non-HTML OSC-8 links are backed by eagerly written temporary copies; the picker writes a copy when selected. The temporary directory is under the configured project root when supplied, otherwise the OS temp directory. Cleanup is attempted on normal interpreter shutdown and is not guaranteed after a crash. CLI HTML links use the optional in-memory listener instead of temporary HTML files. In service mode (`RAGChatService`), bytes are held in the `MarkedDocsStore` in-memory cache and served over HTTP until expiry; the store does not write them to disk. This does not prevent persistence elsewhere; see [In-memory document servers](#in-memory-document-servers-risks-and-safe-operation).
RAG‑LCC’s filtering and detection features are **diagnostic aids**, not security controls.

---

## 🌐 Web / Internet Search

This feature may transmit query data and metadata to external services.
See [LEGAL.md — 🌐 Web / Internet Search: Privacy Warning](LEGAL.md#-web--internet-search--privacy-warning) for full details.

**Mitigations the operator should consider:**

- Set `WEB_SEARCH_MODE = "0"` unless internet search is appropriate.
- Route outbound traffic via a corporate proxy to maintain visibility and control.
- Notify users before enabling this feature in any shared environment.
- Review the search provider's privacy policy and data-retention practices.

Deployment disables the master switch; development settings may differ. Actual use also depends on application and session settings. See also [LEGAL.md — 🌐 Web / Internet Search: Privacy Warning](LEGAL.md#-web--internet-search--privacy-warning).

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
