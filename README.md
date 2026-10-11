# WDBH_GWB

WDBH_GWB calculates the stochastic astrophysical gravitational wave background (AGWB) from populations of compact binaries. The workflow currently supports white dwarf binaries (WDs) and black hole binaries (BHs), and includes population preprocessing, integration over redshift or cosmic time, and plotting and post-processing tools.

The original white dwarf background code was written by [Seppe Staelens](https://github.com/SeppeStaelens/White_Dwarf_AGWB/tree/main). It was subsequently improved by Sophie Hofman; see their [master's thesis](references/master_thesis_Seppe_Staelens.pdf) and [master's thesis](references/master_thesis_Sophie_Hofman.pdf). This project builds on that work by adding black holes to the workflow, refining the parameter-file driven configuration, and extending visualisation and post-processing. The black-hole population data used here are stored under `data/BH` (see [Data](#data)).

## Repository overview

| Path | Contents |
|---|---|
| `param.ini` | Example run and post-processing configuration. See [Parameters](#parameters). |
| `src/GWB.py` | Main calculation driver: loads the selected population and runs bulk, birth bin, and merger bin contributions. |
| `src/SeBa_pre_process.py` | Reads raw SeBa output, derives binary properties, and writes the `Initials_*.txt.gz` tables. |
| `src/Create_z_at_age.py` | Generates the Planck18 age/redshift lookup table. |
| `src/Data_visualisation.py` | Generates spectrum comparisons, redshift contribution charts, and binary count summaries from GWB output. |
| `src/Post_proc.py` | Additional eccentricity diagnostics, model comparisons, uncertainty plots, and initial-population distributions. Uncomment selected recipes in its `main()` function to run them. |
| `tests/` | End-to-end and regression tests using small synthetic populations. |
| `data/` | SeBa populations, preprocessed initial populations, star-formation histories, and cosmology lookup data. |
| `output/GWBs/` | Calculated spectra and intermediate contribution tables. |
| `output/Figures/` | Example spectra and diagnostic/post-processing figures. |

### Data

`data/WD/` and `data/BH/` contain population data arranged as `object/population-synthesis model/alpha prescription/metallicity/`. The population-synthesis choices are `AlphaAlpha` and `GammaAlpha`; the alpha choices are `Alpha1` and `Alpha4`. Raw SeBa tables are named like `<metallicity>_t0aim1m1.dat` (BH) or `.dat.gz` (WD). `SeBa_pre_process.py` derives orbital frequency, chirp mass, inspiral coefficient, maximum frequency, and maximum evolution time, then saves `Initials_<metallicity>.txt.gz` beside the raw file. The committed black-hole data are the SeBa inputs used by this project.

`data/SFRD/` contains tabulated star formation rate density histories (`MZ19`, `LZ19`, `HZ19`, `LZ21`, and `HZ21`). `data/z_at_age.txt.gz` is the age/redshift lookup used by the interpolator. `data/DeciCon10.dat` is an additional data input used by the post-processing utilities.

### Outputs and figures

The main calculation writes files to `output/GWBs/` (or the configured `output_path`). For each selected metallicity, the plain spectrum file contains frequency `f` and background `Om`; `wbirth` and `wmerge` files contain the cumulative spectrum after adding the birth and merger terms. The matching `z_contr`, `z_contr_birth`, and `z_contr_merge` tables record the frequency bin contributions across integration/redshift bins. Filenames encode the SFH selector, frequency and integration bin counts, metallicity, run tag, and object type. Historical result sets are grouped under `output/GWBs/SN24/` and `output/GWBs/HN24/`.

The figures in `output/Figures/` cover these groups:

- `multi_*`: total spectrum across metallicities, with optional WD/BH comparison and detector/sensitivity reference curves.
- `z_contr_*`: stacked contribution to the spectrum by redshift (redshift integration) or corresponding cosmic time bins (time integration).
- `distr_f_*` and `distr_z_*`: binary counts across frequency and redshift, split into the displayed population categories; `TOTAL` denotes the combined metallicity selection.
- `comparison_plot_*` and `uncertainty_plot_*`: comparisons between star formation histories or model tags, including an uncertainty envelope; `_zoom` files show a restricted frequency range.
- `phi_GWB.png`, `phi_curve.png`, and `eccentricity_cloud.png`: eccentric binary harmonic correction and eccentricity evolution diagnostics.
- `initials_*`, `t0_tau_*`, and `nu_max_*`: initial population distributions, respectively chirp mass versus initial GW frequency, formation time versus maximum coalescence time, and maximum GW frequency.
- `SFH1_50_20_example_*` and `HN24_example.png`: example spectra used in the earlier WD background studies.

## Installation and running

Run the workflow on **Linux**. In particular, the `schnell` dependency must be importable in the Linux environment used to run the code. Create and activate the environment from the repository root:

```bash
conda env create -f WDBH_GWB.yml
conda activate WDBH_GWB
```

Run the lookup generator from `src/` because it writes its output relative to the current directory. The other scripts switch to `src/` internally, so run them from the repository root:

```bash
cd src
python Create_z_at_age.py
cd ..
python src/SeBa_pre_process.py param.ini
python src/GWB.py param.ini
python src/Data_visualisation.py param.ini
```

The first script creates the age/redshift lookup as `data/z_at_age.txt`. The repository also includes the compressed lookup `data/z_at_age.txt.gz`, selected by default in `param.ini`; if you regenerate the uncompressed file, set `ri_file = z_at_age.txt` or compress it before running the calculation. Run preprocessing when you need to build or refresh a preprocessed `Initials_*.txt.gz` table from raw SeBa output; set `save_datafile = true` in `param.ini` to write it. Then run `GWB.py` to calculate the spectra. `Data_visualisation.py` reads the configuration and generated outputs. Additional analysis recipes are in `src/Post_proc.py`; edit its `main()` section, removing the leading `#` from a recipe call to enable it and adding `#` again to disable it. The plotting functions' `save` option controls whether they write figures to `output/Figures/`.

## Parameters

`param.ini` is read by the calculation, preprocessing, and `Data_visualisation.py` scripts. Boolean values are `true` or `false` (case insensitive). Frequency bounds are base 10 logarithms of frequency in Hz.

| Section / option | Meaning and accepted values |
|---|---|
| `[integration] log_f_low`, `log_f_high` | Lower and upper `log10(f/Hz)` bounds; numeric, with low < high. |
| `[integration] N_freq` | Number of logarithmic frequency bins; positive integer. |
| `[integration] max_z` | Maximum redshift; positive number. |
| `[integration] N_int` | Number of redshift or cosmic time integration bins; positive integer. |
| `[physics] SFH_num` | Star formation history selector: `1` Madau & Dickinson (2014); `2`–`4` analytic alternatives; `5` constant rate; `6` tabulated SFRD. |
| `[physics] SFH_type` | For `SFH_num = 6`, one of `MZ19`, `LZ19`, `HZ19`, `LZ21`, `HZ21`, matching a file in `data/SFRD/`. Ignored for selectors 1–5. |
| `[physics] loop_over_metallicity` | `true` runs the available metallicity list; `false` runs only `metallicity`. Missing populations are skipped. |
| `[physics] metallicity` | One of `z0001`, `z0005`, `z001`, `z005`, `z01`, `z02`, `z03`. Required when not looping. The tabulated SFRD uses all except `z03`; the WD population at `z0005` may be absent. |
| `[physics] object` | `WD` or `BH`. Selects the compact object data and physics. |
| `[physics] pop_synth` | `AlphaAlpha` or `GammaAlpha`; selects the population synthesis directory. |
| `[physics] alpha` | `Alpha1` or `Alpha4`; selects the alpha prescription directory. |
| `[files] use_data_Seppe` | `true` selects the alternate WD `_Seppe` preprocessed file where available; `false` uses the standard file. Relevant to the main calculation. |
| `[files] ri_file` | Redshift lookup filename under `data/`, normally `z_at_age.txt.gz` or `z_at_age.txt`. |
| `[settings] tag` | Free text used to label output filenames (for example `GA4T`, `AA4T`). Keep it consistent with the data being run. |
| `[settings] integration_mode` | `redshift` or `time`; integrates over redshift bins or cosmic time bins. |
| `[settings] output_path` (optional) | Uncomment the line and set an absolute path or a path relative to `src/` to direct GWB output elsewhere. If omitted, output goes to `../output/GWBs/`. |
| `[settings] save_datafile` | `true` or `false`; whether SeBa preprocessing writes the derived initial population table. |
| `[settings] save_figures` | Present in the sample configuration but currently not read by the scripts; use `save_fig` below for the main calculation. |
| `[settings] save_fig` | `true` or `false`; saves intermediate spectrum plots from the main calculation. |
| `[settings] debug` | `true` or `false`; enables additional diagnostic output. |
| `[settings] test_for_one` | `true` or `false`; prints a sample system for a quick diagnostic run. |
| `[postproc] compare` | `true` or `false`; include a comparison spectrum in `Data_visualisation.py` plots where supported. |
| `[postproc] zfig` | Empty for no individually selected metallicity plot, or comma separated metallicity keys (for example `z01,z02`). |
| `[postproc] save_pp_figures` | `true` or `false`; save figures produced by `Data_visualisation.py`. |
| `[postproc] subfolder` | Empty for current outputs, or `SN24` / `HN24` to load the corresponding historical example output set. |

**Configuration note:** `save_figures` is retained in the sample but is currently unused. Add `save_fig = true` to save intermediate plots from the main calculation. `save_pp_figures` controls `Data_visualisation.py` separately. `output_path` is optional and commented out in the sample. `SeBa_pre_process.py` and `Data_visualisation.py` take the same parameter file, so edit options as needed for each workflow step.

## Calculation notes

The code separates the background into bulk, birth bin, and merger bin contributions. For the count diagnostics, the implementation estimates the number in a redshift/frequency bin as

\[
N(z,f) = 4\pi\,\chi^2(z)\,\Delta\chi(z)\,n(z,f), \qquad
n(z,f) = \sum_k \frac{\psi_k(z)}{S}\,\tau_k(z,f),
\]

where \(\chi\) is comoving distance, \(\Delta\chi\) is the shell width, \(\psi_k\) is the selected star formation rate density for population model \(k\), \(S\) is the SeBa population mass normalization, and \(\tau_k\) is the time a system spends crossing the bin. In the implementation, \(\tau\) is in Myr and is multiplied by \(10^6\) when combined with a rate expressed per year. The `z_contr*` intermediate tables are normalized for numerical convenience; interpret relative contributions consistently within a run.

The thesis PDFs are included for the scientific derivation and context. The equations above summarize the implemented count calculation; consult the [bachelor's thesis](references/bachelor_thesis_Akosh_Kardos.pdf) for the derivation and cite its relevant equations alongside the code. The compact object extension also uses object specific maximum frequency prescriptions (WD contact versus BH innermost stable circular orbit).

## Citation

There is no separate journal article for this software release. For the scientific methods, cite the relevant publications and theses listed above. For the code itself, cite this software repository and version using the metadata in [`CITATION.cff`](CITATION.cff).
