"""Small-sample command-line checks for the repository's main workflows."""

from __future__ import annotations

import configparser
import gzip
from itertools import product
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
METALLICITIES = ("z0001", "z001", "z005", "z01", "z02", "z03")
HISTORY_TYPES = ("LZ19", "MZ19", "HZ19", "LZ21", "HZ21")
POPULATION_MODELS = ("AlphaAlpha", "GammaAlpha")
ALPHA_MODELS = ("Alpha1", "Alpha4")


def _write_config(
    project: Path,
    filename: str,
    *,
    compact_object: str = "BH",
    integration_mode: str = "time",
    sfh_num: int = 1,
    sfh_type: str = "MZ19",
    metallicity: str = "z02",
    pop_synth: str = "GammaAlpha",
    alpha: str = "Alpha4",
    loop_over_metallicity: bool = False,
    use_data_sep: bool = False,
    save_fig: bool = False,
    debug: bool = False,
    test_for_one: bool = False,
    output_path: str = "../output/GWBs/",
    zfig: str = "",
    save_pp_figures: bool = False,
    compare: bool = False,
    tag: str = "ci",
) -> Path:
    config = configparser.ConfigParser()
    config["integration"] = {
        "log_f_low": "-5",
        "log_f_high": "0",
        "N_freq": "3",
        "N_int": "2",
        "max_z": "8",
    }
    config["physics"] = {
        "SFH_num": str(sfh_num),
        "SFH_type": sfh_type,
        "loop_over_metallicity": str(loop_over_metallicity).lower(),
        "metallicity": metallicity,
        "object": compact_object,
        "pop_synth": pop_synth,
        "alpha": alpha,
    }
    config["files"] = {
        "use_data_Seppe": str(use_data_sep).lower(),
        "ri_file": "z_at_age.txt.gz",
    }
    config["settings"] = {
        "tag": tag,
        "integration_mode": integration_mode,
        "output_path": output_path,
        "save_fig": str(save_fig).lower(),
        "save_datafile": "true",
        "debug": str(debug).lower(),
        "test_for_one": str(test_for_one).lower(),
    }
    config["postproc"] = {
        "compare": str(compare).lower(),
        "save_pp_figures": str(save_pp_figures).lower(),
        "zfig": zfig,
    }

    path = project / filename
    with path.open("w", encoding="utf-8") as stream:
        config.write(stream)
    return path


def _run(
    project: Path, command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        command,
        cwd=cwd or project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.fixture(scope="session")
def sample_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create an isolated miniature project so test runs never touch user data."""
    project = tmp_path_factory.mktemp("wd_gwb_sample")
    shutil.copytree(REPOSITORY / "src", project / "src")

    data = project / "data"
    output = project / "output"
    (data / "SFRD").mkdir(parents=True)
    (data / "BH" / "GammaAlpha" / "Alpha4" / "z02").mkdir(parents=True)
    (data / "WD" / "GammaAlpha" / "Alpha4" / "z02").mkdir(parents=True)
    (output / "GWBs").mkdir(parents=True)
    (output / "Figures").mkdir(parents=True)

    for filename in ("z_at_age.txt.gz", "DeciCon10.dat", "ET.txt"):
        shutil.copy2(REPOSITORY / "data" / filename, data / filename)
    for history_type in HISTORY_TYPES:
        shutil.copy2(
            REPOSITORY / "data" / "SFRD" / f"{history_type}_SFRD_allbins.txt.gz",
            data / "SFRD" / f"{history_type}_SFRD_allbins.txt.gz",
        )

    # Five systems per compact-object type keep full workflows fast.
    raw_rows = {
        "BH": (
            "0.1 1.0 5.0 5.0\n"
            "0.2 1.1 6.0 4.0\n"
            "0.4 0.9 7.0 3.0\n"
            "0.6 1.3 4.0 6.0\n"
            "0.8 0.8 8.0 2.0\n"
        ),
        "WD": (
            "0.1 0.5 0.6 0.6\n"
            "0.2 0.55 0.7 0.5\n"
            "0.4 0.45 0.8 0.4\n"
            "0.6 0.65 0.5 0.7\n"
            "0.8 0.6 0.9 0.3\n"
        ),
    }
    for compact_object, rows in raw_rows.items():
        directory = data / compact_object / "GammaAlpha" / "Alpha4" / "z02"
        raw_path = directory / "z02_t0aim1m1.dat"
        if compact_object == "BH":
            raw_path.write_text(rows, encoding="utf-8")
        else:
            with gzip.open(f"{raw_path}.gz", "wt", encoding="utf-8") as stream:
                stream.write(rows)

        config_path = _write_config(
            project,
            f"preprocess_{compact_object}.ini",
            compact_object=compact_object,
        )
        _run(
            project,
            [sys.executable, "src/SeBa_pre_process.py", config_path.name],
        )
        assert (directory / "Initials_z02.txt.gz").exists()

        # Make the same tiny prepared sample available under every selectable
        # population model and metallicity path. This isolates configuration
        # coverage from the full population-synthesis datasets.
        prepared = directory / "Initials_z02.txt.gz"
        for population_model in POPULATION_MODELS:
            for alpha_model in ALPHA_MODELS:
                for metallicity in METALLICITIES:
                    sample_directory = (
                        data
                        / compact_object
                        / population_model
                        / alpha_model
                        / metallicity
                    )
                    sample_directory.mkdir(parents=True, exist_ok=True)
                    sample_path = sample_directory / f"Initials_{metallicity}.txt.gz"
                    if prepared.resolve() != sample_path.resolve():
                        shutil.copy2(prepared, sample_path)
                    shutil.copy2(
                        prepared,
                        sample_directory / f"Initials_{metallicity}_Seppe.txt.gz",
                    )

    # Post_proc.py's legacy diagnostic reads these tables for four models and six z values.
    bh_initials = data / "BH" / "GammaAlpha" / "Alpha4" / "z02" / "Initials_z02.txt.gz"
    sample_table = pd.read_csv(bh_initials)
    for population_model in ("GammaAlpha", "AlphaAlpha"):
        for alpha_model in ("Alpha1", "Alpha4"):
            for metallicity in METALLICITIES:
                directory = data / "BH" / population_model / alpha_model / metallicity
                directory.mkdir(parents=True, exist_ok=True)
                sample_table.to_csv(
                    directory / f"Initials_{metallicity}.txt.gz",
                    index=False,
                    compression="gzip",
                )

    eccentricity_path = data / "BH" / "GammaAlpha" / "Alpha4" / "z02" / "t0aim1m1e.dat.gz"
    with gzip.open(eccentricity_path, "wt", encoding="utf-8") as stream:
        stream.write("0.1 1.0 0.4 5.0 5.0\n0.2 1.1 0.2 6.0 4.0\n")

    return project


def _check_gwb_outputs(
    sample_project: Path,
    integration_mode: str,
    compact_object: str,
    sfh_num: int,
    sfh_type: str,
    metallicity: str,
    pop_synth: str,
    alpha: str,
    tag: str,
) -> None:
    config_path = _write_config(
        sample_project,
        f"{tag}.ini",
        compact_object=compact_object,
        integration_mode=integration_mode,
        sfh_num=sfh_num,
        sfh_type=sfh_type,
        metallicity=metallicity,
        pop_synth=pop_synth,
        alpha=alpha,
        tag=tag,
    )
    _run(sample_project, [sys.executable, "src/GWB.py", config_path.name])

    file_stem = f"SFH{sfh_num}{sfh_type}_3_2_{metallicity}"
    output = sample_project / "output" / "GWBs"
    stage_files = (
        f"{file_stem}_{tag}_{compact_object}.txt",
        f"{file_stem}_wbirth_{tag}_{compact_object}.txt",
        f"{file_stem}_wmerge_{tag}_{compact_object}.txt",
    )
    for result_filename in stage_files:
        result_path = output / result_filename
        assert result_path.exists(), f"Missing GWB stage output: {result_path.name}"
        result = pd.read_csv(result_path)
        assert len(result) == 3
        assert np.isfinite(result["Om"]).all()

    for suffix in ("z_contr", "z_contr_birth", "z_contr_merge"):
        assert (
            output
            / f"SFH{sfh_num}{sfh_type}_3_2_{metallicity}_{suffix}_{tag}_{compact_object}.txt"
        ).exists()


ANALYTIC_CASES = tuple(product(("redshift", "time"), ("BH", "WD"), (1, 2, 3, 4, 5)))


@pytest.mark.parametrize(
    ("integration_mode", "compact_object", "sfh_num"),
    ANALYTIC_CASES,
    ids=lambda value: str(value),
)
def test_gwb_small_population_analytic_histories(
    sample_project: Path,
    integration_mode: str,
    compact_object: str,
    sfh_num: int,
) -> None:
    """Run both integration modes and compact-object types for analytic SFHs."""
    pop_synth = POPULATION_MODELS[(sfh_num + (compact_object == "WD")) % 2]
    alpha = ALPHA_MODELS[(sfh_num + (integration_mode == "time")) % 2]
    tag = f"ci-{integration_mode}-{compact_object}-sfh{sfh_num}"
    _check_gwb_outputs(
        sample_project,
        integration_mode,
        compact_object,
        sfh_num,
        "MZ19",
        "z02",
        pop_synth,
        alpha,
        tag,
    )


TABULATED_CASES = tuple(
    product(("redshift", "time"), ("BH", "WD"), HISTORY_TYPES, METALLICITIES)
)


@pytest.mark.parametrize(
    ("integration_mode", "compact_object", "sfh_type", "metallicity"),
    TABULATED_CASES,
    ids=lambda value: str(value),
)
def test_gwb_small_population_tabulated_histories(
    sample_project: Path,
    integration_mode: str,
    compact_object: str,
    sfh_type: str,
    metallicity: str,
) -> None:
    """Exercise every tabulated SFH, metallicity, integration, and object option."""
    tag = f"ci-{integration_mode}-{compact_object}-{sfh_type}-{metallicity}"
    config_path = _write_config(
        sample_project,
        f"{tag}.ini",
        compact_object=compact_object,
        integration_mode=integration_mode,
        sfh_num=6,
        sfh_type=sfh_type,
        metallicity=metallicity,
        tag=tag,
    )
    history_index = HISTORY_TYPES.index(sfh_type)
    metallicity_index = METALLICITIES.index(metallicity)
    pop_synth = POPULATION_MODELS[(history_index + metallicity_index) % 2]
    alpha = ALPHA_MODELS[(history_index + metallicity_index + (compact_object == "WD")) % 2]
    _check_gwb_outputs(
        sample_project,
        integration_mode,
        compact_object,
        6,
        sfh_type,
        metallicity,
        pop_synth,
        alpha,
        tag,
    )


def test_gwb_metallicity_loop(sample_project: Path) -> None:
    """Exercise the parameter-file option that loops over all metallicities."""
    tag = "ci-metallicity-loop"
    config_path = _write_config(
        sample_project,
        "metallicity_loop.ini",
        compact_object="BH",
        integration_mode="redshift",
        sfh_num=6,
        sfh_type="MZ19",
        loop_over_metallicity=True,
        tag=tag,
    )
    result = _run(sample_project, [sys.executable, "src/GWB.py", config_path.name])
    assert "Skipping z0005 because:" in result.stdout
    for metallicity in METALLICITIES:
        result = (
            sample_project
            / "output"
            / "GWBs"
            / f"SFH6MZ19_3_2_{metallicity}_wmerge_{tag}_BH.txt"
        )
        assert result.exists(), f"Metallicity loop did not produce {result.name}"


def test_optional_gwb_settings(sample_project: Path) -> None:
    """Cover one-row mode, Seppe data naming, and optional GWB figure output."""
    tag = "ci-options"
    config_path = _write_config(
        sample_project,
        "optional_settings.ini",
        compact_object="BH",
        integration_mode="time",
        sfh_num=1,
        use_data_sep=True,
        save_fig=True,
        debug=True,
        test_for_one=True,
        tag=tag,
    )
    _run(sample_project, [sys.executable, "src/GWB.py", config_path.name])
    assert (
        sample_project / "output" / "GWBs" / f"SFH1MZ19_3_2_z02_{tag}_BH.txt"
    ).exists()
    for contribution in ("", "wbirth_", "wmerge_"):
        image = (
            sample_project
            / "output"
            / "Figures"
            / f"GWB_SFH1MZ19_3_2_{contribution}{tag}_BH.png"
        )
        assert image.exists(), f"Expected saved figure {image.name}"


def test_gwb_custom_output_directory(sample_project: Path) -> None:
    """Ensure output_path directs result files to a configured directory."""
    custom_output = sample_project / "output" / "custom_gwbs"
    custom_output.mkdir()
    tag = "ci-custom-path"
    config_path = _write_config(
        sample_project,
        "custom_output.ini",
        compact_object="WD",
        output_path="../output/custom_gwbs/",
        tag=tag,
    )
    _run(sample_project, [sys.executable, "src/GWB.py", config_path.name])
    assert (custom_output / f"SFH1MZ19_3_2_z02_{tag}_WD.txt").exists()


def test_data_visualisation_command(sample_project: Path) -> None:
    """Exercise spectrum, contribution, and system-count plotting from GWB outputs."""
    tag = "ci-time-BH-1"
    config_path = _write_config(
        sample_project,
        "visualisation.ini",
        compact_object="BH",
        integration_mode="time",
        sfh_num=1,
        zfig="z02",
        save_pp_figures=True,
        tag=tag,
    )
    _run(sample_project, [sys.executable, "src/GWB.py", config_path.name])
    _run(
        sample_project,
        [sys.executable, "src/Data_visualisation.py", config_path.name],
    )
    figures = sample_project / "output" / "Figures"
    expected_figures = (
        f"multi_3_2_SFH1MZ19_{tag}_BH_chosen_z.png",
        f"multi_3_2_SFH1MZ19_{tag}_BH_1z.png",
        f"z_contr_3_2_SFH1MZ19_{tag}_z02_BH.png",
        f"distr_f_3_2_SFH1MZ19_{tag}_z02_BH.png",
        f"distr_z_3_2_SFH1MZ19_{tag}_z02_BH.png",
    )
    for filename in expected_figures:
        assert (figures / filename).exists(), f"Missing visualization {filename}"


def test_data_visualisation_metallicity_loop(sample_project: Path) -> None:
    """Exercise the combined SFH6 plots for the all-metallicity setting."""
    config_path = _write_config(
        sample_project,
        "visualisation_loop.ini",
        compact_object="BH",
        integration_mode="redshift",
        sfh_num=6,
        sfh_type="MZ19",
        loop_over_metallicity=True,
        tag="ci-metallicity-loop",
    )
    _run(
        sample_project,
        [sys.executable, "src/Data_visualisation.py", config_path.name],
    )


def test_data_visualisation_compares_population_types(sample_project: Path) -> None:
    """Exercise the BH plot option that overlays the matching WD spectrum."""
    tag = "ci-compare"
    output = sample_project / "output" / "GWBs"
    frequencies = np.logspace(-5, 0, 3)
    pd.DataFrame({"f": frequencies, "Om": [1e-12, 2e-12, 3e-12]}).to_csv(
        output / f"SFH1MZ19_3_2_z02_wmerge_{tag}_BH.txt", index=False
    )
    pd.DataFrame({"f": frequencies, "Om": [2e-12, 3e-12, 4e-12]}).to_csv(
        output / f"SFH1MZ19_3_2_z01_wmerge_{tag}_WD.txt", index=False
    )
    contribution = {"z": [1.0, 2.0], "T": [100.0, 200.0]}
    for index in range(3):
        contribution[f"freq_{index}"] = [1.0, 2.0]
        contribution[f"freq_{index}_num"] = [1e6, 2e6]
    for suffix in ("z_contr", "z_contr_birth", "z_contr_merge"):
        pd.DataFrame(contribution).to_csv(
            output / f"SFH1MZ19_3_2_z02_{suffix}_{tag}_BH.txt", index=False
        )

    config_path = _write_config(
        sample_project,
        "visualisation_compare.ini",
        compact_object="BH",
        sfh_num=1,
        compare=True,
        tag=tag,
    )
    _run(
        sample_project,
        [sys.executable, "src/Data_visualisation.py", config_path.name],
    )


def test_post_processing_diagnostics(sample_project: Path) -> None:
    """Exercise Post_proc's eccentricity, population, and comparison plots."""
    script = sample_project / "postproc_checks.py"
    script.write_text(
        """import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))

ns = runpy.run_path("Post_proc.py")
population = pd.DataFrame({
    "e": [0.4, 0.2, 0.3, 0.5, 0.1],
    "a": [1.0, 1.1, 0.9, 1.3, 0.8],
    "m1": [5.0, 6.0, 7.0, 4.0, 8.0],
    "m2": [5.0, 4.0, 3.0, 6.0, 2.0],
})
ns["compute_phi_gwb_harmonic"](
    population, n_pole=2, n_e_steps=5, n_fr_bins=4
)
ns["eccentricity_cloud"](population, n_pole=2, n_e_steps=5)
ns["phi_curve"](0.4, n_pole=2)
ns["density_initials"]("GammaAlpha", "Alpha4")
ns["density_t0_tau"]("GammaAlpha", "Alpha4")
assert ns["phi_n"](2, 0.4) > 0
assert ns["e_at_fp_ratio"](2.0, 0.4) >= 0

output = Path("../output/GWBs")
frequencies = np.logspace(-5, 2, 80)
for tag in ("GA1T", "GA4T", "AA1T", "AA4T"):
    for metallicity in ("z0001", "z001", "z005", "z01", "z02", "z03"):
        spectrum = pd.DataFrame({
            "f": frequencies,
            "Om": (1 + len(tag) + len(metallicity)) * 1e-12 * frequencies ** (2 / 3),
        })
        spectrum.to_csv(
            output
            / f"SFH6MZ19_100_20_{metallicity}_wmerge_{tag}_BH.txt",
            index=False,
        )
ns["compare_plot"]("tag", "AA4T", "MZ19")
ns["gwb_uncertainty_plot"]("MZ19", "GA1T", "MZ19", "GA4T", save=False)
""",
        encoding="utf-8",
    )
    _run(
        sample_project,
        [sys.executable, "../postproc_checks.py"],
        cwd=sample_project / "src",
    )


def test_redshift_lookup_generator(sample_project: Path) -> None:
    """Run the lookup-table generator in the isolated sample project."""
    _run(
        sample_project,
        [sys.executable, "Create_z_at_age.py"],
        cwd=sample_project / "src",
    )
    lookup = pd.read_csv(sample_project / "data" / "z_at_age.txt")
    assert len(lookup) == 10_000
    assert lookup["Age (Myr)"].is_monotonic_increasing
    assert np.isfinite(lookup["Redshift"]).all()
