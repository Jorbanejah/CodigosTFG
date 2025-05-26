import numpy as np

def metodo_potencia(C, iter=1000, tol=1e-6):
    n, _ = C.shape
    b_k = np.random.rand(n)
    b_k=b_k/np.linalg.norm(b_k)  # Vector inicial simbólico de modulo la unidad
    for _ in range(iter):
        b_k1 = np.dot(C,b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm == 0:
            raise ValueError("La matriz tiene un autovalor 0")
        
        b_k = b_k1 / b_k1_norm
        
        if np.linalg.norm(np.dot(C,b_k)-b_k1_norm*b_k)<tol:
            break
    autovector=b_k
    autovalor = np.dot(autovector.conj().conj(), np.dot(C, autovector))
   
    return autovalor,autovector
def metodo_potencia_todos_autovalores(C, iter=100, tol=1e-6): 
    #se hace por un método de deflación y se hace de manera simbolica. Debido a que el método de la potencia me da el autvalor dominante de la matriz C
    # una vez calculado yo quito ese autovalor y empiezo de nuevo con el método.
    
    n, _ = C.shape
    autovalores = []
    autovectores= []
    C_defl = C.copy()  # Copiamos la matriz original
    
    for _ in range(n):
        [autovalor, autovector] = metodo_potencia(C_defl, iter, tol) #cojo el método de la potencia
        autovalores.append(autovalor)#metemos al final de la lista de autovalores el autovalor calculado por el metodo de la potencia
        autovectores.append(autovector)
        # Construimos la matriz de deflación: C' = C - autovalor * (autovector * autovector.Transpuesto)
        autovector = autovector / np.linalg.norm(autovector)  # Normalizamos autovector
        C_defl = C_defl - autovalor*np.outer(autovector,autovector.conj())
    
    return autovalores, autovectores