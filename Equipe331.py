import numpy as np
import matplotlib.pyplot as plt
from problimite import problimite

def p(x):
    return -1/x  

def q(x):
    return np.zeros_like(x)  

def r(x):
    return (-1.6/x**4)  

# Solution exacte
def y_exact(x, c, d):
    return (c/x**2) - (c/0.4 - d)*np.log(x) + (c/0.4 - d)*np.log(0.9)


A = np.array([
    [1/0.9**2 - (1/0.4)*np.log(0.9), np.log(0.9)],
    [1/1**2, np.log(1) - np.log(0.9)]
])
b = np.array([0, 0])
c, d = np.linalg.solve(A, b)


a, b_val = 0.9, 1.0
alpha, beta = 0.0, 0.0


plt.figure(figsize=(10, 6), dpi=100)
x_exact = np.linspace(a, b_val, 500)
plt.plot(x_exact, y_exact(x_exact, c, d), 'k-', linewidth=2, label='Solution exacte')


for h, color, style in zip([1/30, 1/100], ['blue', 'red'], ['--', '-.']):
    N = int((b_val-a)/h) - 1
    x_num = np.linspace(a, b_val, N+2)
    y_num = problimite(h, p(x_num[1:-1]), q(x_num[1:-1]), r(x_num[1:-1]), a, b_val, alpha, beta)
    plt.plot(x_num, y_num, style, color=color, linewidth=1.5, 
             label=f'Approximation h={h:.3f}')

plt.xlabel('Distance à l\'axe (x)', fontsize=12)
plt.ylabel('Température (y)', fontsize=12)
plt.title('Distribution de température entre les cylindres', fontsize=14)
plt.legend(fontsize=10, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('solution_comparison.png', bbox_inches='tight')
plt.show()

h_values = [1/10, 1/30, 1/100, 1/300, 1/1000]
errors = []

for h in h_values:
    N = int((b_val-a)/h) - 1
    x_num = np.linspace(a, b_val, N+2)
    y_num = problimite(h, p(x_num[1:-1]), q(x_num[1:-1]), r(x_num[1:-1]), a, b_val, alpha, beta)
    error = np.max(np.abs(y_num - y_exact(x_num, c, d)))
    errors.append(error)

plt.figure(figsize=(10, 6), dpi=100)
plt.loglog(h_values, errors, 'bo-', markersize=8, linewidth=2, label='Erreur observée')
plt.loglog(h_values, [h**2 for h in h_values], 'k--', linewidth=1.5, label='Référence O(h²)')

plt.xlabel('Pas de discrétisation (h)', fontsize=12)
plt.ylabel('Erreur maximale E(h)', fontsize=12)
plt.title('Convergence de la méthode des différences finies', fontsize=14)
plt.legend(fontsize=10, framealpha=0.9)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xticks(h_values, [f'{h:.0e}' for h in h_values])
plt.tight_layout()
plt.savefig('error_convergence.png', bbox_inches='tight')
plt.show()