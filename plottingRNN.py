import os
import re
import matplotlib.pyplot as plt

hidden_dims = [64, 128, 256]
epochs_list = [3, 5, 10]
results = {}

# Ensure the folder exists
os.makedirs('resultsRNN', exist_ok=True)

# Extract validation accuracies from result files
for h in hidden_dims:
    for e in epochs_list:
        filename = os.path.join('resultsRNN', f'results_{h}_{e}.txt')
        try:
            with open(filename, 'r') as f:
                text = f.read()
                match = re.search(r'Best validation accuracy is:\s*([\d.]+)', text)
                if match:
                    acc = float(match.group(1))
                    results[(h, e)] = acc
        except FileNotFoundError:
            print(f"{filename} not found.")

# Plotting
plt.figure(figsize=(10, 6))
for h in hidden_dims:
    accs = [results.get((h, e), 0) for e in epochs_list]
    plt.plot(epochs_list, accs, marker='o', label=f'Hidden Dim = {h}')

plt.title('Validation Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = os.path.join('resultsRNN', 'validation_accuracy_plot.png')
plt.savefig(plot_path)

print(f"Plot saved to {plot_path}")
