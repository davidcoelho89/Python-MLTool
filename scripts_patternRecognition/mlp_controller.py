# Treinamento rede MLP - Controlador

from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

a1 = 1
a2 = 1
a3 = 1
a4 = 1
a5 = 1
a6 = 1

b1 = 1
b2 = 1
b3 = 1
b4 = 1
b5 = 1
b6 = 1

# Gera sinal de Referencia
Ref = np.zeros((1,600000))
Ref[1,0:99999] = 1800
Ref[1,100000:199999] = 1400
...

# Gera sinal carga 60s. 10kHz. 600.000 pontos
Cl = np.zeros(1,600000)
Cl[1,130000:170000] = 20;
Cl[1,230000:270000] = 50;
...
# Inicializa vetor de velocidade
w = np.zeros(1,600000)

# Inicializa vetor de corrente
i = np.zeros(1,600000)

# Inicializa vetor de tensao
V = np.zeros(1,600000)

# Inicializa vetor de erros
error = np.zeros((1,2))

# Inicializa MLP
MLPcontroller = MLPRegressor(hidden_layer_sizes=(6), random_state=1, max_iter=200)

for n in range(2,600000):
    
    # Calcula corrente
    i[1,n] = a1*i[1,n-1] + a2*i[1,n-2] + a3*V[1,n-1] + a4*V[1,n-2] + a5*Cl[1,n-1] + a6*Cl[1,n-2]
    
    # Calcula velocidade
    w[1,n] = b1*w[1,n-1] + b2*w[1,n-2] + b3*V[1,n-1] + b4*V[1,n-2] + b5*Cl[1,n-1] + b6*Cl[1,n-2]
    
    # Gera vetor de entrada para rede
    X = [i[1,n], i[1,n-1] w[1,n] w[1,n-1] Ref[1,n] Ref[1,n-1] V[1,n] V[1,n-1]]
    
    # Prediz Valor
    V[1,n] = MLPcontroller.predict(X)
    
    # Calcula erro de predição
    error[1,n] = ref[1,n] - w[1,n]
    
    # Atualiza pesos
    MLPcontroller.update(X,ref[1,n])

end


