import re

_ACTIVE_BANNED_CONFIG = "_STRICT_BANNED"
_STRICT_BANNED = {
    "BANNED": [
        "ssn",
        "social security number",
        "tax id",
        "tax identification number",
        "passport number",
        "id card number",
        "id number",
        "driver license number",
        "driving licence number",
        "phone number",
        "mobile number",
        "telephone number",
        "street address",
        "home address",
        "billing address",
        "account number",
        "iban",
        "bic",
        "bank account",
        "routing number",
        "private-key",
        "secret-key",
        "api-key",
        "auth token",
        "jwt",
        "session token",
        "credit card number",
        "card number",
        "cvv",
        "ccv",
        "card expiry",
        "mm/yy",
        "account balance",
        "transaction history",
        "salary amount",
        "health insurance card",
        "health insurance number",
        "diagnosis",
        "disability",
        "genetic",
        "biometric",
        "race",
        "ethnicity",
        "belief",
        "political opinion",
        "trade union",
        "sexual orientation",
        "criminal record",
        "conviction",
        "offence",
        "court order",
        "warrant",
        "subpoena",
        "legal claim",
        "data subject",
        "data controller",
        "data processor",
        "risk score",
        "credit score",
        "fraud score",
        "background check",
        "voting behavior",
        "employment decision",
        "loan decision",
        "minor",
        "child",
        "under 18",
        "date of birth",
        "dob",
        "mother's maiden name",
        "security question",
        "explosive",
        "weapon",
        "hack",
        "malware",
        "exploit",
        "backdoor",
        "steal",
        "extort",
        "buffer overflow",
        "bomb",
    ],
}

_ACTIVE_MASKING_CONFIG = "_STRICT_MASKING_REGEXES"
# Strict masking regexes configuration
# - Each rule contains: pattern, mask (action), enabled (bool), priority (int), desc (human description)
# - Runtime should: validate/compile patterns on startup, sort by priority descending, apply first-match or all-match per policy

_STRICT_MASKING_REGEXES = {
    "MASKING_REGEXES": {
        # Credit card formats: strict 4-4-4-4 with optional separators
        "CREDIT_CARD_STRICT": {
            "pattern": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
            "mask": "mask_credit_card",
            "enabled": True,
            "priority": 10,
            "desc": "Mask common 4-4-4-4 credit card formats (preserve separators)",
        },
        # Loose long-digit sequences that resemble card numbers (may over-match)
        "CREDIT_CARD_LOOSE": {
            "pattern": r"(?<![0-9a-fA-F])(?:\d[ -]*?){13,19}(?![0-9a-fA-F])",
            "mask": "mask_credit_card",
            "enabled": True,
            "priority": 11,
            "desc": "Mask long digit sequences that look like credit cards (preserve separators); excludes hex strings",
        },
        # Fallback plain digits only (no separators) — higher priority to catch raw numbers
        "CREDIT_CARD_PLAIN_FALLBACK": {
            "pattern": r"(?<![0-9a-fA-F])(?:\d{13,19})(?![0-9a-fA-F])",
            "mask": "mask_credit_card",
            "enabled": True,
            "priority": 12,
            "desc": "Fallback for long digit sequences without separators; excludes hex strings",
        },
        # CVV-like 3 or 4 digit sequences — disabled by default due to high false-positive risk
        "CVV_THREE_FOUR": {
            "pattern": r"(?<!\d)(?:\d{3}|\d{4})(?!\d)",
            "mask": "[CVV]",
            "enabled": False,
            "priority": 13,
            "desc": "CVV-like 3 or 4 digit sequences (disabled by default; enable if needed)",
        },
        # Email masking: preserve domain, mask local part via runtime logic (use named groups)
        "EMAIL": {
            "pattern": r"(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            "mask": "mask_email",
            "enabled": True,
            "priority": 20,
            "desc": "Mask email local part leaving domain visible",
        },
        # US Social Security Number formats
        "SSN_DASH": {
            "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
            "mask": "[SSN]",
            "enabled": True,
            "priority": 30,
            "desc": "US SSN with dashes",
        },
        "SSN_PLAIN": {
            "pattern": r"\b\d{9}\b",
            "mask": "[SSN]",
            "enabled": True,
            "priority": 31,
            "desc": "US SSN plain 9 digits",
        },
        # IBAN detection (disabled by default because of international variety)
        "IBAN": {
            "pattern": r"\b[A-Z]{2}[0-9A-Z]{13,34}\b",
            "mask": "[IBAN]",
            "enabled": False,
            "priority": 35,
            "desc": "IBAN-like sequences (disabled by default; enable if you handle IBANs)",
        },
        # IPv4 masking: Standard to mask only last octet at runtime
        "IPV4_LAST_OCTET": {
            "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "mask": "mask_ip_last_octet",
            "enabled": True,
            "priority": 40,
            "desc": "Mask last octet of IPv4 addresses",
        },
        # IPv6 detection (disabled by default; enable if you expect IPv6)
        "IPV6": {
            "pattern": r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b",
            "mask": "[IPV6]",
            "enabled": False,
            "priority": 41,
            "desc": "IPv6 addresses (disabled by default; enable if needed)",
        },
        # MAC addresses
        "MAC": {
            "pattern": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
            "mask": "[MAC]",
            "enabled": True,
            "priority": 50,
            "desc": "Mask MAC addresses",
        },
        # UUIDs (v1-v5 pattern)
        "UUID": {
            "pattern": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
            "mask": "[UUID]",
            "enabled": True,
            "priority": 60,
            "desc": "Mask UUIDs",
        },
        # JWT-like tokens (base64url header.payload.signature)
        "JWT": {
            "pattern": r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
            "mask": "mask_jwt",
            "enabled": True,
            "priority": 70,
            "desc": "Mask JWT-like tokens",
        },
        # AWS access key IDs
        "AWS_ACCESS_KEY": {
            "pattern": r"\b(?:AKIA|ASIA)[0-9A-Z]{12,20}\b",
            "mask": "[AWS]",
            "enabled": True,
            "priority": 40,
            "desc": "Mask AWS access key IDs (flexible length)",
        },
        # GCP service account key fingerprint (disabled by default)
        "GCP_SERVICE_ACCOUNT_KEY": {
            "pattern": r"\b[0-9a-fA-F]{40}\b",
            "mask": "[GCP_KEY]",
            "enabled": False,
            "priority": 81,
            "desc": "GCP-like service account key (disabled by default; tune if needed)",
        },
        # Long base64-like sequences — aggressive, may over-match
        "LONG_BASE64_LIKE": {
            "pattern": r"\b[A-Za-z0-9/+=]{40,}\b",
            "mask": "[SECRET]",
            "enabled": True,
            "priority": 90,
            "desc": "Mask long base64-like secrets (may over-match; tune or disable if aggressive)",
        },
        # Key=value style secrets (password=..., api_key: ...)
        "KV_PASSWORDS": {
            "pattern": r"(?P<key>\bpassword\b|\bpasswd\b|\bsecret\b|\bapi_key\b|\bapikey\b|\baccess_token\b)(?P<sep>\s*[:=]\s*)(?P<val>[^,\s;\"']+)",
            "mask": "mask_kv_secret",
            "enabled": True,
            "priority": 100,
            "desc": "Mask key=value style secrets (password=..., api_key: ...)",
        },
        # Windows-style Password=... fragments (case-sensitive keys included)
        "WINDOWS_PWD": {
            "pattern": r"(?P<key>\bPwd\b|\bPassword\b|\bpwd\b)(?P<sep>\s*[:=]\s*)(?P<val>[^,\s;\"']+)",
            "mask": "mask_kv_secret",
            "enabled": True,
            "priority": 101,
            "desc": "Mask Windows-style Password=... fragments",
        },
        # Simple password fallback (case-insensitive)
        "PASSWORD_SIMPLE": {
            "pattern": r"(?i)\b(pass(word)?|pw)\b\s*[:=]\s*([^\s,;]+)",
            "mask": "[PASSWORD]",
            "enabled": True,
            "priority": 110,
            "desc": "Simple password key/value fallback",
        },
        # Connection string password extraction (disabled by default)
        "CONNECTION_STRING_PASSWORD": {
            "pattern": r"(?i)(?:User\s*Id|Uid|User|Username|Password|Pwd)\s*=\s*([^;]+)",
            "mask": "[REDACTED_CONN]",
            "enabled": False,
            "priority": 111,
            "desc": "Connection-string style key=value pairs (disabled by default; enable if you parse connection strings)",
        },
        # E.164 phone numbers (disabled by default)
        "PHONE_E164": {
            "pattern": r"\+?[1-9]\d{1,14}",
            "mask": "[PHONE]",
            "enabled": False,
            "priority": 120,
            "desc": "E.164 phone numbers (disabled by default; enable if needed)",
        },
        # 9-digit sequences that may be routing numbers or SSNs (disabled by default)
        "ROUTING_ABA": {
            "pattern": r"\b\d{9}\b",
            "mask": "[BANK_ROUTING_OR_SSN]",
            "enabled": False,
            "priority": 130,
            "desc": "9-digit sequences that may be routing numbers or SSNs (disabled by default; careful with false positives)",
        },
        # Example custom token rule (disabled by default)
        "CUSTOM_TOKEN_EXAMPLE": {
            "pattern": r"\bCUSTOM-[A-Za-z0-9]{20}\b",
            "mask": "[CUSTOM_TOKEN]",
            "enabled": False,
            "priority": 140,
            "desc": "Example custom token rule (disabled by default)",
        },
    }
}

# ---------------------------------------------------------------------------
# Hard-blocked content patterns — checked unconditionally by WebRetriever;
# cannot be disabled via configuration.
# (compiled_pattern, human-readable reason)
# ---------------------------------------------------------------------------
HARDBLOCK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Child sexual abuse material
    (
        re.compile(
            r"(?i)\b(csam|child\s+(porn|sex|abuse\s+material|exploitation)|"
            r"loli(ta)?\s+porn|underage\s+(sex|porn|nude))\b"
        ),
        "hard block \u2014 CSAM",
    ),
    # Synthesis / manufacture of weapons of mass destruction
    (
        re.compile(
            r"(?i)\b(synthesize|synthesis|manufacture|make|produce|create|build)\b"
            r".{0,60}"
            r"\b(sarin|novichok|vx\b|mustard\s+gas|nerve\s+agent|"
            r"ricin|anthrax\s+spore|botulinum|weaponized\s+pathogen|"
            r"dirty\s+bomb|nuclear\s+device|radiological\s+weapon|"
            r"plastic\s+explosive|c4\b|semtex|rdx\b|tatp|hmtd)\b"
        ),
        "hard block \u2014 WMD / explosive synthesis",
    ),
    # Synthesis / manufacture of illegal drugs (actionable instructions)
    (
        re.compile(
            r"(?i)\b(synthesize|synthesis|manufacture|cook|make|produce)\b"
            r".{0,60}"
            r"\b(fentanyl|methamphetamine|meth\b|heroin|crack\s+cocaine|"
            r"carfentanil|opioid\s+analogue)\b"
        ),
        "hard block \u2014 controlled substance synthesis",
    ),
    # Human trafficking
    (
        re.compile(
            r"(?i)\b(buy|sell|traffic|smuggle|recruit)\b.{0,50}"
            r"\b(human|person|child|girl|boy|worker)\b.{0,50}"
            r"\b(slave|trafficking|forced\s+labor|sex\s+work)\b"
        ),
        "hard block \u2014 human trafficking",
    ),
    # Hire / contract violence
    (
        re.compile(
            r"(?i)\b(hire|contract|find|pay).{0,40}"
            r"\b(hitman|assassin|killer|murderer|to\s+(kill|murder|assassinate))\b"
        ),
        "hard block \u2014 solicitation of violence",
    ),
]

# ---------------------------------------------------------------------------
# Injection / attack detection patterns — gated by block_on_injection config.
# (compiled_pattern, human-readable reason)
# ---------------------------------------------------------------------------
INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?i)\bignore\b.{0,60}\b(previous|prior|above|all)\b.{0,60}"
            r"\b(instruction|prompt|rule|directive|constraint)s?\b"
        ),
        "prompt injection \u2014 ignore instructions",
    ),
    (re.compile(r"(?i)\byou\s+are\s+now\b"), "persona override \u2014 'you are now'"),
    (re.compile(r"(?i)\bact\s+as\b.{0,30}\b(an?\b|the)\b"), "persona override \u2014 'act as'"),
    (re.compile(r"(?i)\bsystem\s*:\s"), "system-role injection"),
    (
        re.compile(r"(?i)\b(disregard|forget|override)\b.{0,40}\b(instruction|prompt|rule)s?\b"),
        "instruction override",
    ),
    (
        re.compile(
            r"(?i)\b(show|print|output|reveal|leak|send|exfiltrate)\b.{0,50}"
            r"\b(password|passphrase|api.?key|secret|credential|token)\b"
        ),
        "credential exfiltration",
    ),
    (
        re.compile(
            r"(?i)\b(drop|delete|truncate|alter)\b.{0,30}\b(table|database|schema|index)\b"
        ),
        "SQL destructive command",
    ),
    (re.compile(r"[;|&`$]{3,}"), "shell injection characters"),
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), "control characters"),
]


