import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generamos datos aletaorias simulados 
np.random.seed(42)  #Utilizamos una semilla para poder después reproducir los mismos resultados
data = np.random.rand(100, 5) # 100 observaciones con 5 variables
# Datos simulados
pca = PCA() #Inicializamos el PCA
pca.fit(data)  # Ajusta los datos: calcula los componentes principales y la varianza explicada
# Obtenemos la varianza explicada por cada componente principal

# Plotemos el diagrama de sedimentación/ grafica  scree

plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
plt.title('Diagrama de sedimentación')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Proporción de Varianza Explicada')
plt.show()

