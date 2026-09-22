# E1-C Latent Probe Paired Statistical Comparison

## Scientific question and comparison

Did the E1-C anatomy/background reconstruction constraint induce preferential pathology accessibility in Z_d relative to Z_a?

**Delta = Z_d probe Dice - Z_a probe Dice**

## Frozen inputs

| Latent | Input file | SHA256 |
|---|---|---|
| Z_a | `outputs/frozen/e1c_probes/anatomy/evaluation/per_case_metrics.csv` | `d1328d3b800490cf56ee44532d185c9fd2e76cb9ca470fa713df809607212880` |
| Z_d | `outputs/frozen/e1c_probes/disease/evaluation/per_case_metrics.csv` | `f68053782d926051c06ea2b6619652c3ee95f9f366f0065b163764d32515185b` |

## Analysis provenance and protocol

- Analysis implementation commit: `56ca6715266aad2b82637d95067f80da1128fc24`.
- `git_dirty=false` during analysis.
- Analysis environment: Python 3.11.15; NumPy 2.2.6; SciPy 1.17.1.
- 270 validation cases; target-positive cases only for each region.
- Positive counts: NETC 114; SNFH 270; ET 198; RC 230; WT 270; TC 200.
- Paired difference rows = 1282.
- Paired percentile bootstrap of mean Z_d-minus-Z_a Dice; 10,000 bootstrap resamples; bootstrap seed 20260917.
- 95% confidence interval using 2.5th and 97.5th percentiles.
- Two-sided Wilcoxon signed-rank test: `zero_method="wilcox"`, `correction=False`, `method="auto"`.
- All-zero paired differences -> p=1.0.
- Holm step-down correction across six regional tests.
- Win/tie tolerance: 1e-8.

## Results

Za and Zd are mean target-positive case Dice for the Z_a and Z_d probes, respectively.

| Region | N | Za | Zd | Delta | 95% CI | raw p | Holm p | Zd wins | ties | Za wins |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| NETC | 114 | 0.17182311921032237 | 0.17971451103093977 | +0.007891391820617412 | [-0.004305326335630972, 0.02060149974366364] | 0.4670986372763647 | 1.0 | 51 | 18 | 45 |
| SNFH | 270 | 0.6260963008282017 | 0.6142332881482111 | -0.01186301267999052 | [-0.016048813021431366, -0.007815355240298368] | 1.3961313322584605e-09 | 6.980656661292303e-09 | 89 | 0 | 181 |
| ET | 198 | 0.2755190539238457 | 0.27402708732162606 | -0.0014919666022196357 | [-0.01068713546529497, 0.007332847044774391] | 0.8617082872091948 | 1.0 | 90 | 15 | 93 |
| RC | 230 | 0.5511099311376355 | 0.5549697924868466 | +0.003859861349211196 | [-0.0022111780472218725, 0.009738083751011718] | 0.3174100638079952 | 1.0 | 114 | 10 | 106 |
| WT | 270 | 0.6778241740953591 | 0.6648113936737732 | -0.013012780421585948 | [-0.016832274300805118, -0.009463007519952953] | 9.669624997792169e-13 | 5.801774998675302e-12 | 79 | 0 | 191 |
| TC | 200 | 0.29886336458905133 | 0.30108454795219586 | +0.0022211833631445186 | [-0.007103460891692521, 0.011640424728302604] | 0.8102021120963385 | 1.0 | 93 | 13 | 94 |

## Scientific interpretation

Under identical frozen latent-probe conditions, the E1-C anatomy/background reconstruction constraint did not induce preferential pathology accessibility in Z_d relative to Z_a. No evaluated region showed a statistically significant Z_d advantage after Holm correction. Z_a showed significantly higher probe performance for SNFH and WT after Holm correction. For NETC, ET, RC, and TC, there was insufficient evidence of a difference.

- These results measure relative pathology accessibility under the frozen probe protocol.
- They do not establish semantic disentanglement.
- Do not describe E1-C as "failed disentanglement".
- The appropriate conclusion is that the E1-C constraint was insufficient to induce the desired preferential pathology accessibility in Z_d.
- E1-C Z_a must not be called a healthy or normative latent based on these results.

## Frozen output SHA256

Files are relative to `outputs/frozen/e1c_probe_paired_statistics/`.

| File | SHA256 |
|---|---|
| `analysis_provenance.json` | `e9921db17f628ddd54639ba4c13126fb2e8fd30e5bf3012ce1630cbb6074a514` |
| `analysis_summary.json` | `1f19e5af478ff07958b46de187d532c287d5475cea146a07416e68a6fee4c9bd` |
| `paired_differences.csv` | `db185a30f5cb73c3fa6ca216e8caa5a0f64588ac0b01de2d2c2b4ecea9da0a14` |
| `paired_statistics.csv` | `c5174df6276ebf3f85c5932d707f671ccdaf9984a6b348f1bf8666a910c3fdc1` |
| `paper_table.csv` | `1ce59efe299d03b2a8e93517f4d6855ef9379676ee4bd1120a36cdded26f0819` |
