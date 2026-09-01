# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting page:

https://github.com/ttqiao1617/MiniLLM-RT/security/advisories/new

Include the affected commit, reproduction steps, impact and any suggested
mitigation. Avoid including real credentials or personal data in the report.

## Trust boundaries

MiniLLM-RT is a local research runtime, not a hardened multi-tenant service.
Treat model repositories, ONNX files, TensorRT engine files, model manifests
and benchmark inputs as executable or otherwise security-sensitive content.

- Use the pinned Qwen model and full commit revision in the provided scripts.
- Do not load models, ONNX files or TensorRT engines from untrusted sources.
- Build TensorRT engines locally for the target GPU and verify their origin.
- Build containers from an explicitly reviewed image digest.
- Do not commit generated environment snapshots before reviewing them.

The environment collector redacts hostname, GPU UUID, process lists and local
absolute paths by default. `--include-host-identifiers` deliberately opts back
into collecting that information; such output should be treated as private.

## Supported version

Security fixes are applied to the latest commit on `main`. Older snapshots and
locally generated artifacts are not supported.
