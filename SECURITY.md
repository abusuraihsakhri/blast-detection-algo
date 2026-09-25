# Security Policy

## Reporting a Vulnerability

The maintainers of the **Leukemic Blast Detection Clinical & Research Pipeline** take code security and patient privacy seriously. If you discover a security vulnerability, path traversal risk, data leakage issue, or denial-of-service vulnerability in this codebase, please report it privately.

**Please do NOT disclose vulnerabilities publicly in GitHub Issues.**

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
   - Image uploads are bounded by `MAX_UPLOAD_SIZE = 25MB`.
   - Decompression bomb guard: `Image.MAX_IMAGE_PIXELS = 50,000,000` to prevent memory exhaustion from zip/image bombs.
   - Bounded annotation payload lists (`max_length=500`).
3. **MIME Validation:** Strictly restricts uploaded file types to safe image formats (`image/jpeg`, `image/png`, `image/bmp`, `image/tiff`, `image/webp`).
4. **Access Control & Specimen Privacy:** The temporary sandbox upload directory is strictly private and NOT mounted as a static public route.
5. **CORS & HTTP Security Headers:** Restricts CORS origins strictly to localhost/127.0.0.1 and enforces `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY`.
