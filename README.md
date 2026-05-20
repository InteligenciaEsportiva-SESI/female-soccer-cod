# Fast in a Straight Line, Inefficient in Turns

**An efficiency paradox in knee extensor strength, sprint speed, and change of direction deficit in youth female soccer players.**

This repository hosts the statistical analysis code, generated figures, and supporting scripts for the manuscript of the same name. It is intended to support transparent peer review and replication: the manuscript's headline regression analyses, robustness checks, and sensitivity arms can all be re-executed from the source files here against the de-identified dataset deposited on Zenodo.

## Authors

1. **Aldo Seffrin** — SESI-SP — Sport Intelligence, São Paulo, Brazil. ORCID: [0000-0001-8229-8565](https://orcid.org/0000-0001-8229-8565)
2. **Ronaldo Kobal de Oliveira Alves Cardoso** — Brazilian Football Confederation (CBF), Rio de Janeiro, Brazil. ORCID: [0000-0003-0243-2133](https://orcid.org/0000-0003-0243-2133)
3. **Paulo Azevedo** — Federal University of São Paulo (UNIFESP), Santos, Brazil. ORCID: [0000-0002-4293-9214](https://orcid.org/0000-0002-4293-9214)
4. **Rodrigo Fernandes** — SESI-SP — Sport Intelligence, São Paulo, Brazil
5. **Cesar Cavinato Cal Abad** — SESI-SP — Sport Intelligence, São Paulo, Brazil *(corresponding author — cesar.abad@sesisp.org.br)*. ORCID: [0000-0003-4988-5076](https://orcid.org/0000-0003-4988-5076)

## Study summary

A cross-sectional study of 43 youth female soccer players from a national sports excellence programme (Under-15 and Under-17 squads). A 22-athlete subset additionally underwent concentric isokinetic dynamometry of the knee extensors and flexors, providing the mechanical predictors for the central analyses. Multiple linear regression with continuous standardised predictors (and a Bayesian sensitivity arm with weakly informative priors) tested whether relative knee extensor peak torque, the hamstring-to-quadriceps (H:Q) ratio, and the reactive strength index from a 30-cm drop jump related to 20-m linear sprint time and the relative change of direction deficit (CODD) in the same direction.

**Headline result.** Higher relative concentric knee extensor torque was associated with both faster 20-m sprint times (β = −0.106 per SD, R² = 0.612) and, simultaneously, a greater change of direction deficit (β = +5.32 per SD, R² = 0.398). The Bayesian sensitivity arm preserved both directions at high posterior probability (probability of direction = 0.9998 and 0.988, respectively). A complementary 4-predictor sensitivity regression with menstrual status entered as a binary covariate did not alter the paradoxical direction.

## Data availability

The de-identified analysis dataset is deposited on Zenodo with a citable DOI under a restricted-access licence (to protect the privacy of the minor participants):

**DOI: [10.5281/zenodo.20274968](https://doi.org/10.5281/zenodo.20274968)**

Access may be granted by the corresponding author upon reasonable request for purposes consistent with the original ethical consent framework. The raw dataset is not committed to this repository; the notebooks expect the dataset at `data/Database_Soccer_ana02.csv` after the user populates the `data/` directory from the Zenodo record.

## Repository contents

```
female-soccer-cod/
├── notebooks/                    # Analysis pipeline
│   ├── data_prep.py                  # Shared data loader + derived variables
│   ├── _common.py                    # Shared constants + helpers
│   ├── reliability.py                # Test–retest reliability (ICC, CV%)
│   ├── descriptives.py               # Descriptive statistics + sample profile
│   ├── inferential.py                # Pearson correlations + multiple regression
│   ├── figures.py                    # Publication figures (TIFF 300 dpi)
│   ├── robustness_check.py           # Continuous OLS + Bayesian sensitivity
│   │                                 # + menstrual-status 4-predictor arm
│   ├── cycle_data_cross_reference.py # Joins iso subsample with the daily
│   │                                 # wellness form (PII-safe outputs only)
│   └── results/                      # Generated outputs (idempotent re-runs)
│       ├── reliability_*.{txt,csv}
│       ├── descriptives_*.{txt,csv}
│       ├── inferential_*.{txt,csv}
│       ├── robustness_check_*.{txt,csv}
│       ├── robustness_sensitivity_menstrual_*.{txt,csv}
│       ├── cycle_data_cross_reference.txt
│       └── cycle_data_iso_match_anon.csv
├── figures/                      # Publication figures (TIFF 300 dpi)
│   ├── Figure1_Paradox.tiff
│   ├── Figure2_Mechanism.tiff
│   └── Figure3_Typology.tiff
├── requirements.txt              # Pinned Python dependencies
├── LICENSE                       # MIT
└── README.md                     # This file
```

## Reproducing the analyses

1. Clone this repository.
2. Request the dataset from the corresponding author (or the Zenodo deposit) and place the de-identified CSV at `data/Database_Soccer_ana02.csv`.
3. Create the Python environment from `requirements.txt`:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run the analysis pipeline (Python modules, idempotent):
   ```bash
   cd notebooks
   ../.venv/bin/python reliability.py
   ../.venv/bin/python descriptives.py
   ../.venv/bin/python inferential.py
   ../.venv/bin/python cycle_data_cross_reference.py
   ../.venv/bin/python robustness_check.py
   ../.venv/bin/python figures.py
   ```
   Outputs are written to `notebooks/results/` and `figures/`. Re-running the pipeline produces byte-identical files (verified via MD5).

## Analysis modules

The pipeline is broken into small, single-purpose modules. Numbered execution order matches the manuscript's narrative:

- **`reliability.py`** — test–retest reliability for the field battery (Table 1). Reports ICC(3,1) (two-way mixed, consistency, single rater), within-subject CV%, and SEM.
- **`descriptives.py`** — descriptive statistics for the field (n = 43) and isokinetic (n = 22) samples (Tables 2 and 3): mean, SD, t-based 95% CI, Shapiro–Wilk normality.
- **`inferential.py`** — Pearson correlations (Table 4) and continuous multiple linear regression with all three mechanical predictors retained and z-standardised (Tables 5 and 6, replacing the original stepwise selection), plus a descriptive median-split supplementary comparison (legacy Table 7 / Supplementary Table S1).
- **`robustness_check.py`** — continuous OLS (matching `inferential.py`) plus a **weakly informative Bayesian arm** in PyMC (NUTS, 4 chains × 2000 draws + 1000 tune, target_accept = 0.95, seed = 42 for full reproducibility), and a **4-predictor sensitivity arm** that adds menstrual status as a binary covariate (menstruating Yes / No at testing). The side-by-side comparison is written to `results/robustness_sensitivity_menstrual_*.{txt,csv}`.
- **`cycle_data_cross_reference.py`** — joins the isokinetic subsample (n = 22) with the daily coaching-staff wellness form responses to obtain concurrent menstrual status and contraceptive use for each athlete. PII (athlete names) is contained to a local, gitignored CSV in `data/`; the tracked outputs use anonymised `iso_id` (1–22) only.
- **`figures.py`** — regenerates the three publication figures (TIFF 300 dpi): Figure 1 (paradox), Figure 2 (mechanism, median-split comparison), Figure 3 (athlete typology).

Convergence diagnostics, prior calibration notes, and the full posterior summaries for the Bayesian arms are reproduced in the manuscript's Statistical Output Appendix.

## Citation

The manuscript is under peer review at *Science and Medicine in Football*; this citation block will be updated upon acceptance.

For the dataset alone:

> Seffrin, A., Kobal de Oliveira Alves Cardoso, R., Azevedo, P., Fernandes, R., & Cavinato Cal Abad, C. (2026). *Female youth soccer — knee extensor strength, sprint speed and CODD dataset* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.20274968>

## Ethics

The study was approved by the Research Ethics Committee of the Centro Universitário FIEO (UNIFIEO), Osasco, Brazil (Opinion 5.021.137; CAAE 51970721.7.0000.5435). All participants provided written assent in age-appropriate language, in addition to written informed consent obtained from their legal guardians; the assent process was conducted independently of parental presence to mitigate institutional coercion.

## License

Code and documentation: MIT — see [`LICENSE`](LICENSE).
Dataset: restricted-access via Zenodo (see *Data availability* above).
