import re
import matplotlib.pyplot as plt

# Ścieżka do pliku z logami
log_file = "output/V2/CustomEffNet_90ep-BS16_LR25e-4-LS5e-2-MA-DR50/log-CustomEffNet_90ep-BS16_LR25e-4-LS5e-2-MA.txt"

# Listy do przechowywania wyników
train_losses = []
val_losses = []

# Wzorzec regex do wyciągania danych
epoch_pattern = re.compile(
    r"Epoch\s+\d+/\d+,\s+Training loss:\s+([0-9.]+),\s+Validation loss:\s+([0-9.]+)"
)

# Wczytywanie i parsowanie pliku
with open(log_file, 'r') as f:
    for line in f:
        match = epoch_pattern.search(line)
        if match:
            train_loss = float(match.group(1))
            val_loss = float(match.group(2))
            train_losses.append(train_loss)
            val_losses.append(val_loss)

# Sprawdzenie, czy dane zostały wczytane
if not train_losses:
    print("Nie znaleziono danych w pliku logów.")
    exit()

# Rysowanie wykresów
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Strata treningowa')
plt.plot(val_losses, label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Funkcja straty')
plt.title('Funkcja straty w czasie treningu')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
