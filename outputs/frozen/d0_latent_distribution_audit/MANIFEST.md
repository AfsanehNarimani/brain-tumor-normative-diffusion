# D0-A — E1-C Za Latent Distribution Audit

## Scientific purpose

Train-split-only numerical audit of the frozen E1-C Za anatomy/background representation before latent diffusion.

## Audit scope and sampling

- This was **NOT model training**. No optimizer, backpropagation, or model update occurred.
- Validation data were not used for statistics. The E1-C source model was frozen and in eval mode.
- The audit used deterministic pre-augmentation training patches: exactly 10 patches per training case, comprising 7 foreground-centered and 3 random patches.
- Biological sample size: **1080 train cases**; **10,800 total patches**. Patches are repeated samples within cases, not independent subjects.
- Each patch contributed a `[256, 12, 12, 12]` latent: **4,777,574,400 total latent values**.

## Source provenance

| Source | Commit or SHA256 |
|---|---|
| E1-C checkpoint SHA256 | `cf5a04a2d845cecb45a0ea87d617abad0afab9ad4264abce8342a6045563ee53` |
| Train split SHA256 | `581dc6326a22cb0b9582fa249b8e13fd948597a60c5f454cb5ade0493fa7c18e` |
| D0-A scientific audit commit | `6572f50b42b7fc10aaf3c56fa8d0a4c9aaf2d63c` |
| Normalization-contract builder commit | `93a65d699efb41aba53f9276a71945459c23256b` |

## Locked normalization decision for D0-B

**Fixed per-channel standardization** using the frozen `d0_za_normalization_contract.json`:

```text
(Za[c] - mean[c]) / population_std[c]
```

- 256 channel-specific means/stds, derived only from D0-A train data.
- Population standard deviation, **ddof=0**.
- Validation and inference reuse the same fixed values without recomputation.
- No per-case normalization, no per-patch normalization, and no clipping.

## Scientific evidence and interpretation

| Quantity | Approximate value or range |
|---|---|
| Global Za mean | -0.030693 |
| Global Za std | 1.625129 |
| Channel mean range | -6.186331 to +5.370539 |
| Channel std range | 0.440706 to 2.800078 |

Case-level and patch-level aggregate scale was comparatively stable; therefore substantial heterogeneity was primarily inter-channel. This evidence motivates the locked per-channel standardization decision. It does **not** establish that this preprocessing improves diffusion performance.

Za denotes the **anatomy/background representation**. These numerical distribution measurements do not establish semantic disentanglement or justify a healthy or normative interpretation of Za.

## Immutable artifact SHA256

The following eight scientific artifacts were transferred byte-for-byte from the scientific server run. Paths are relative to `outputs/frozen/d0_latent_distribution_audit/`. Preserve their bytes unchanged; the hashes below identify the frozen artifacts.

| Artifact | SHA256 |
|---|---|
| `analysis_provenance.json` | `dd11667d47b7d920211a8f992abf985de200621544afab744065011654505aed` |
| `d0_za_normalization_contract.json` | `bca5c0f8f1617d73fc4d0850fe5b6f45e4b2bcbacbb4018f1d5d825de49ad792` |
| `latent_case_statistics.csv` | `727be5e8a2d2b418b8f955ed979b074d19fce3b9f718f8eb4a67f404f8b0a13a` |
| `latent_channel_statistics.csv` | `5a962dc620122974c23e95a5d31cd5ce4577c67c9f03fa61c02acfafdb8457e6` |
| `latent_distribution_summary.json` | `425fdf553dd1d723d7d60869c611962813ec6e04fdd9e781aa158d875aa580f1` |
| `latent_global_statistics.json` | `cadc59b27810397cf4794c5c52b8abc214141f875c485d6def63ff16e83c06db` |
| `latent_patch_statistics.csv` | `387612f14c53cdbaabf48aa2b511b219da5e005067ea2ef8261722b59db737ee` |
| `patch_manifest.csv` | `70eed5eccd2ea5227a4895a4737b4bc0a359d4447c9275876d321f5faf1c0fbe` |
