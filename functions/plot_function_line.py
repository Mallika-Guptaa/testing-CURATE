def plot_function_line(folder: str, input1: str, input2: str, output1: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import json
    import tempfile
    import os

    faasr_log("Reading workflow config")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_cfg:
        tmp_cfg_path = tmp_cfg.name

    try:
        faasr_get_file(local_file=tmp_cfg_path, remote_folder=folder, remote_file=input2)
        with open(tmp_cfg_path) as f:
            config = json.load(f)
    finally:
        os.unlink(tmp_cfg_path)

    controls = config.get("nodes", {}).get("plot_function_line", {})

    faasr_log("Reading function points CSV")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_in:
        tmp_in_path = tmp_in.name

    try:
        faasr_get_file(local_file=tmp_in_path, remote_folder=folder, remote_file=input1)
        df = pd.read_csv(tmp_in_path)
    finally:
        os.unlink(tmp_in_path)

    faasr_log(f"Plotting {len(df)} points for y = 20x + 33")

    fig, ax = plt.subplots()
    ax.plot(df["x"], df["y"], marker="o", linestyle="-")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("y = 20x + 33")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        fig.savefig(tmp_out_path)
        plt.close(fig)
        faasr_put_file(local_file=tmp_out_path, remote_folder=folder, remote_file=output1)
        faasr_log(f"Saved plot to {output1}")
    finally:
        os.unlink(tmp_out_path)
