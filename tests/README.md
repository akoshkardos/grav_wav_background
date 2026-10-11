# Tests

The end-to-end tests use small temporary projects, so they do not overwrite the
repository's input data or generated thesis figures. The test setup preprocesses
five synthetic binaries for each compact-object type, then runs the calculation
and plotting scripts against those small populations.

The GWB checks cover both integration modes, BH and WD populations, analytic
SFH selectors 1-5, tabulated selector 6 with every available history and
metallicity, and the metallicity-loop and optional-output settings. The
unavailable WD `z0005` sample is checked for a clean skip. Separate checks
exercise SeBa preprocessing, Data_visualisation, Post_proc, and the redshift
lookup generator.

Create the environment and run the suite from the repository root:

```bash
conda env create -f WDBH_GWB.yml
conda activate WDBH_GWB
pytest -q
```

The GitHub Actions workflow runs the same suite on pushes, pull requests, and
manual dispatch. These checks verify that the workflows run, produce expected
files, and return finite spectra. They are smoke and regression checks; they do
not compare results against a trusted numerical baseline or replace a full
thesis-data reproduction.

*Pytest script was made with Codex.
