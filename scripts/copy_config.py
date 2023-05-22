from shutil import copy

import yaml

files = {
        "config.yaml": "config.yaml",
        "Snakefile": "Snakefile",
        "scripts/resolve_network.py": "solve_network.py",
        }

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake("copy_config")

    basepath = (
        snakemake.config["results_dir"] + "/" + snakemake.config["run"] + "/configs/"
    )

    for f, name in files.items():
        copy(f, basepath + name)

    with open(basepath + "config.yaml", "w") as yaml_file:
        yaml.dump(
            snakemake.config,
            yaml_file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )