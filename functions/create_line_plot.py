def create_line_plot(folder: str, input1: str, input2: str, output1: str) -> None:
    """Read CSV data points, create line plot with markers, save as PNG."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json
    import os

    faasr_log("Starting create_line_plot")

    csv_local = "function_points_local.csv"
    faasr_get_file(local_file=csv_local, remote_folder=folder, remote_file=input1)

    config_local = "workflow_config_local.json"
    faasr_get_file(local_file=config_local, remote_folder=folder, remote_file=input2)

    with open(config_local, "r") as f:
        config = json.load(f)

    plot_config = config.get("nodes", {}).get("create_line_plot", {})
    faasr_log(f"Plot config: {plot_config}")

    df = pd.read_csv(csv_local)
    faasr_log(f"Loaded {len(df)} data points")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['x'], df['y'], marker='o', linestyle='-', linewidth=2, markersize=6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('y = 2x + 3')
    ax.grid(True, alpha=0.3)

    local_output = "function_line_plot_local.png"
    fig.savefig(local_output, dpi=100, bbox_inches='tight')
    plt.close(fig)

    faasr_log(f"Created line plot: {local_output}")

    faasr_put_file(local_file=local_output, remote_folder=folder, remote_file=output1)

    for f_path in [csv_local, config_local, local_output]:
        if os.path.exists(f_path):
            os.remove(f_path)

    faasr_log("create_line_plot complete")
