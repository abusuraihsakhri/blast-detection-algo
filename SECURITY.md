# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, path traversal risk, data leakage issue, or denial-of-service vulnerability in this codebase, please report it privately.

**Please do not disclose vulnerabilities publicly in GitHub Issues.**

### How to Report
- Email: **abusuraihsakhri@gmail.com**
- Subject: `[SECURITY] Blast Algo Vulnerability Report`
- Include:
  1. Description of the vulnerability and OWASP 2025 category.
  2. Step-by-step reproduction instructions or proof-of-concept payload.
  3. Affected endpoints or modules.
  4. Proposed remediation code, if available.

Reports will be acknowledged within 48 hours, followed by a remediation timeline.

---

## Security Architecture & Controls

This repository implements the following baseline defenses:

1. **Path Traversal Defenses:** All file paths sourced from client uploads or SQLite database records are validated strictly using `validate_path_within()` before reading, copying, or writing.
2. **Denial-of-Service (DoS) Protections:**
   - Image uploads are bounded by `settings.max_upload_bytes` (default 25 MB).
   - Decompression bomb guard: `settings.max_upload_pixels` (default 50 Megapixels) to prevent memory exhaustion from zip/image bombs.
   - Bounded annotation payload lists (`max_length=5000`).
3. **MIME Validation:** Restricts uploaded file types to safe image formats (`image/jpeg`, `image/png`, `image/bmp`, `image/webp`).
4. **Access Control & File Privacy:** The temporary sandbox upload directory is stored in an internal directory and not mounted as a static public route.
5. **CORS & HTTP Security Headers:** Restricts CORS origins strictly to localhost/127.0.0.1 and enforces `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and Content Security Policy.
