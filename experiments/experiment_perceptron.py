import torch
import matplotlib.pyplot as plt

from models.perceptron import Perceptron

# Data generation for perceptron problem
torch.manual_seed(42)

# Set dimensions
m, d = 10, 2

# Synthetic data
X_perceptron = torch.randn(m,d) # generate random data
w_opt        = torch.randn(d)   # generate random optimal solution
y_perceptron = torch.sign(
    X_perceptron.matmul(w_opt)) # generate solution-consistent labels

# Test this out on the synthetic data
model = Perceptron(data = (X_perceptron, y_perceptron))
w_perceptron, t = model.perceptron_algorithm(X_perceptron, y_perceptron)
acc_perceptron = (torch.sign(X_perceptron.matmul(w_perceptron)) == y_perceptron).float().mean()

print("w_perceptron :", w_perceptron)
print("t :", t)

print(f"Accuracy: {acc_perceptron:.2f} in {t} rounds.")

margin = model.perceptron_margin(X_perceptron,w_opt)
print(f"Margin: {margin:.2f}. The perce-ptron algorithm is guaranteed to terminate in {1/(margin**2):.2f} rounds.")

# Plot the classifier. Is it close to the the ground truth?
plt.scatter(X_perceptron[:,0], X_perceptron[:,1], c=y_perceptron)
X_range = torch.linspace(-2,2, 50)
plt.plot(X_range, -w_perceptron[0]/w_perceptron[1]*X_range) # estimated
plt.plot(X_range, -w_opt[0]/w_opt[1]*X_range) # optimal
plt.legend(['Perceptron Algorithm', 'Ground Truth'])
plt.grid()