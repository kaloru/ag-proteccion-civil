"""
Algoritmo Genético — Asignación de Recursos de Protección Civil
Región Occidente de México (Jalisco, Colima, Michoacán, Nayarit)

Ejecutar con:
    python ag_proteccion_civil.py

Requiere: pandas, numpy, matplotlib
"""

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  Parámetros del AG
# ─────────────────────────────────────────────
PRESUPUESTO       = 10_000_000   # 10 millones de pesos
COSTO_INTERVENCION = 80_000      # costo fijo por municipio
TAM_POBLACION     = 100
NUM_GENERACIONES  = 200
K_TORNEO          = 5
NUM_CORRIDAS      = 5


# ─────────────────────────────────────────────
#  Cargar y preprocesar el dataset
# ─────────────────────────────────────────────
def cargar_datos(ruta_csv):
    df = pd.read_csv (ruta_csv)
    return df


# Mapeos de puntaje para cada variable categórica
peso_zona = {
    'B': 1,
    'C': 2,
    'D': 3
}

peso_inundacion = {
    'Muy bajo': 1,
    'Bajo':     2,
    'Medio':    3,
    'Alto':     4,
    'Muy alto': 5
}

peso_vulnerabilidad = {
    'Muy bajo': 1,
    'Bajo':     2,
    'Medio':    3,
    'Alto':     4,
    'Muy alto': 5
}


def calcular_puntaje_municipio(fila):
    """
    Calcula el puntaje de un municipio usando la fórmula ponderada.
    Los que no tienen ARM reciben un bono del 30%.
    """
    ps = peso_zona.get(fila['zona_sismica_CFE'], 1)
    pi = peso_inundacion.get(fila['grado_peligro_inundacion_CENAPRED'], 1)
    pv = peso_vulnerabilidad.get(fila['grado_vulnerabilidad_social_CENAPRED'], 1)

    # Si no tiene Atlas de Riesgos le damos más prioridad
    bono_arm = 1.3 if fila['tiene_ARM'] == 'No' else 1.0

    puntaje = (0.40 * ps + 0.35 * pi + 0.25 * pv) * bono_arm
    return puntaje


# ─────────────────────────────────────────────
#  Función de aptitud
# ─────────────────────────────────────────────
def calcular_aptitud(cromosoma, puntajes):
    """
    Suma los puntajes de los municipios seleccionados.
    Si se pasa de los 10 millones devuelve 0 (solución inválida).
    """
    indices_sel = [i for i, gen in enumerate(cromosoma) if gen == 1]
    presupuesto_actual = len(indices_sel) * COSTO_INTERVENCION

    # Checamos que no se pase de los 10 millones
    if presupuesto_actual > PRESUPUESTO:
        return 0

    return sum(puntajes[i] for i in indices_sel)


# ─────────────────────────────────────────────
#  Operadores genéticos
# ─────────────────────────────────────────────
def seleccion_torneo(poblacion, aptitudes, k=K_TORNEO):
    """
    Torneo de tamaño k: elige k individuos al azar y gana el mejor.
    """
    candidatos = random.sample(range(len(poblacion)), k)
    ganador = max(candidatos, key=lambda i: aptitudes[i])
    return poblacion[ganador][:]  # devuelve copia


def cruzar_padres(padre1, padre2):
    """
    Cruzamiento de un punto: elige un punto aleatorio y mezcla los dos padres.
    Produce dos hijos.
    """
    n = len(padre1)
    punto_corte = random.randint(1, n - 1)

    hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
    hijo2 = padre2[:punto_corte] + padre1[punto_corte:]

    return hijo1, hijo2


def mutar(cromosoma, tasa_mutacion):
    """
    Mutación bit a bit: cada gen se invierte con probabilidad tasa_mutacion.
    """
    for i in range(len(cromosoma)):
        if random.random() < tasa_mutacion:
            cromosoma[i] = 1 - cromosoma[i]
    return cromosoma


# ─────────────────────────────────────────────
#  Algoritmo genético completo
# ─────────────────────────────────────────────
def ejecutar_ag(puntajes, tasa_mutacion, semilla=None):
    """
    Ejecuta el AG completo y devuelve:
      - mejor cromosoma encontrado
      - aptitud de ese cromosoma
      - historial de mejor aptitud por generación (para la gráfica)
    """
    n = len(puntajes)

    if semilla is not None:
        random.seed(semilla)

    # Población inicial aleatoria
    poblacion = [
        [random.randint(0, 1) for _ in range(n)]
        for _ in range(TAM_POBLACION)
    ]

    mejor_cromosoma_global = None
    mejor_aptitud_global   = -1
    historial_fitness      = []

    for generacion in range(NUM_GENERACIONES):
        # Evaluamos toda la población
        aptitudes = [calcular_aptitud(c, puntajes) for c in poblacion]

        # Guardamos el mejor de esta generación
        mejor_idx = max(range(len(aptitudes)), key=lambda i: aptitudes[i])
        if aptitudes[mejor_idx] > mejor_aptitud_global:
            mejor_aptitud_global   = aptitudes[mejor_idx]
            mejor_cromosoma_global = poblacion[mejor_idx][:]

        historial_fitness.append(mejor_aptitud_global)

        # Construimos la nueva generación
        nueva_poblacion = []
        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo1, hijo2 = cruzar_padres(padre1, padre2)
            nueva_poblacion.append(mutar(hijo1, tasa_mutacion))
            nueva_poblacion.append(mutar(hijo2, tasa_mutacion))

        # Nos quedamos exactamente con TAM_POBLACION individuos
        poblacion = nueva_poblacion[:TAM_POBLACION]

    return mejor_cromosoma_global, mejor_aptitud_global, historial_fitness


# ─────────────────────────────────────────────
#  Experimentos: 5 corridas por tasa
# ─────────────────────────────────────────────
def correr_experimentos(puntajes, tasas, semillas):
    resultados = {}

    for tasa in tasas:
        print(f"\n{'='*50}")
        print(f"Tasa de mutación: {tasa}")
        print(f"{'='*50}")

        corridas = []
        for i, semilla in enumerate(semillas):
            cromosoma, aptitud, historial = ejecutar_ag(puntajes, tasa, semilla)
            presupuesto_usado = sum(cromosoma) * COSTO_INTERVENCION
            num_municipios    = sum(cromosoma)

            print(f"  Corrida {i+1} (semilla={semilla}): "
                  f"fitness={aptitud:.4f}, "
                  f"presupuesto=${presupuesto_usado:,}, "
                  f"municipios={num_municipios}")

            corridas.append({
                'cromosoma':        cromosoma,
                'aptitud':          aptitud,
                'presupuesto_usado': presupuesto_usado,
                'municipios':       num_municipios,
                'historial':        historial,
                'semilla':          semilla
            })

        aptitudes = [c['aptitud'] for c in corridas]
        print(f"  Promedio fitness: {sum(aptitudes)/len(aptitudes):.4f}")
        print(f"  Mejor fitness:    {max(aptitudes):.4f}")

        resultados[tasa] = corridas

    return resultados


# ─────────────────────────────────────────────
#  Análisis de la mejor solución
# ─────────────────────────────────────────────
def analizar_mejor_solucion(df, cromosoma):
    """
    Imprime un resumen de qué municipios quedaron en la mejor solución.
    """
    seleccionados = df[[bool(g) for g in cromosoma]].copy()

    print("\n--- Análisis de la mejor solución ---")
    print(f"Municipios atendidos: {len(seleccionados)}")
    print(f"Presupuesto usado:    ${len(seleccionados) * COSTO_INTERVENCION:,}")
    print("\nPor estado:")
    print(seleccionados['estado'].value_counts().to_string())
    print("\nPor zona sísmica:")
    print(seleccionados['zona_sismica_CFE'].value_counts().to_string())
    print("\nPor ARM:")
    print(seleccionados['tiene_ARM'].value_counts().to_string())
    print("\nTop 10 municipios por puntaje:")
    top10 = seleccionados.sort_values('puntaje', ascending=False).head(10)
    cols = ['estado', 'municipio', 'zona_sismica_CFE',
            'grado_peligro_inundacion_CENAPRED',
            'grado_vulnerabilidad_social_CENAPRED',
            'tiene_ARM', 'puntaje']
    print(top10[cols].to_string(index=False))


# ─────────────────────────────────────────────
#  Gráfica de convergencia
# ─────────────────────────────────────────────
def graficar_convergencia(resultados, tasas):
    """
    Grafica el historial del mejor fitness por generación,
    promediando las 5 corridas de cada tasa.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colores = {0.01: '#1f77b4', 0.05: '#d62728'}
    estilos = {0.01: '-', 0.05: '--'}

    for tasa in tasas:
        corridas = resultados[tasa]
        # Promediamos los historiales de las 5 corridas
        historiales = np.array([c['historial'] for c in corridas])
        promedio    = historiales.mean(axis=0)
        maximo      = historiales.max(axis=0)

        generaciones = range(1, NUM_GENERACIONES + 1)

        ax.plot(generaciones, promedio,
                color=colores[tasa],
                linestyle=estilos[tasa],
                linewidth=2,
                label=f'Tasa mutación = {tasa} (promedio 5 corridas)')

        # Banda de la mejor corrida
        ax.fill_between(generaciones, promedio, maximo,
                        color=colores[tasa], alpha=0.15)

    ax.set_xlabel('Generación', fontsize=12)
    ax.set_ylabel('Mejor fitness acumulado', fontsize=12)
    ax.set_title('Convergencia del AG — Asignación de Recursos de Protección Civil\n'
                 'Región Occidente de México', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('convergencia_ag.png', dpi=150)
    print("\nGráfica guardada como: convergencia_ag.png")
    plt.show()


# ─────────────────────────────────────────────
#  Punto de entrada
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Ruta al CSV — cambiar si está en otra carpeta
    RUTA_CSV = 'municipios_occidente_riesgo_CENAPRED.csv'

    print("Cargando datos...")
    df = cargar_datos(RUTA_CSV)
    print(f"Dataset cargado: {len(df)} municipios en {df['estado'].nunique()} estados.")

    # Calculamos el puntaje de cada municipio una sola vez
    df['puntaje'] = df.apply(calcular_puntaje_municipio, axis=1)
    puntajes = df['puntaje'].values

    # Tasas a comparar y semillas para reproducibilidad
    tasas   = [0.01, 0.05]
    semillas = [10, 20, 30, 40, 50]

    # Corremos los experimentos
    resultados = correr_experimentos(puntajes, tasas, semillas)

    # Identificamos la mejor solución global
    mejor_tasa     = None
    mejor_cromosoma = None
    mejor_aptitud_global = -1

    for tasa, corridas in resultados.items():
        for corrida in corridas:
            if corrida['aptitud'] > mejor_aptitud_global:
                mejor_aptitud_global = corrida['aptitud']
                mejor_cromosoma      = corrida['cromosoma']
                mejor_tasa           = tasa

    print(f"\n{'='*50}")
    print(f"MEJOR SOLUCIÓN GLOBAL")
    print(f"  Tasa de mutación: {mejor_tasa}")
    print(f"  Fitness:          {mejor_aptitud_global:.4f}")
    print(f"{'='*50}")

    analizar_mejor_solucion(df, mejor_cromosoma)

    # Generamos la gráfica de convergencia
    graficar_convergencia(resultados, tasas)
