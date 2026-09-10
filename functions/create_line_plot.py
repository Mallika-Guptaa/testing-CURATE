def create_line_plot(folder: str, input1: str, input2: str, output1: str) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    faasr_log("Reading input CSV file")
    local_input = "function_points.csv"
    faasr_get_file(local_file=local_input, remote_folder=folder, remote_file=input1)

    faasr_get_file(local_file="workflow-config.json", remote_folder=folder, remote_file=input2)

    df = pd.read_csv(local_input)

    faasr_log("Creating line plot with markers")
    plt.figure(figsize=(10, 6))
    plt.plot(df['x'], df['y'], marker='o', linestyle='-', color='blue', markersize=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Function: y = 2x + 3')
    plt.grid(True, alpha=0.3)

    local_output = "function_line_plot.png"
    plt.savefig(local_output, dpi=100, bbox_inches='tight')
    plt.close()

    faasr_put_file(local_file=local_output, remote_folder=folder, remote_file=output1)

    faasr_log(f"Saved plot to {output1}")
