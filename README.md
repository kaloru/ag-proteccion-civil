# Algoritmo Genético — Asignación de Recursos de Protección Civil
**Materia:** Inteligencia Artificial / Algoritmos Bioinspirados  
**Región:** Occidente de México (Jalisco, Colima, Michoacán, Nayarit)

## Descripción
Implementación de un Algoritmo Genético para optimizar la asignación
de un presupuesto de $10,000,000 MXN entre 267 municipios del occidente
de México, priorizando los de mayor peligro sísmico, inundación y
vulnerabilidad social según datos del CENAPRED.

## Archivos
| Archivo | Descripción |
|---------|-------------|
| `ag_proteccion_civil.py` | Código principal del AG |
| `municipios_occidente_riesgo_CENAPRED.csv` | Dataset CENAPRED |

## Requisitos
```bash
pip install pandas numpy matplotlib
```

## Cómo correrlo
```bash
python ag_proteccion_civil.py
```
Al terminar genera la gráfica `convergencia_ag.png` en la misma carpeta.

## Resultados principales
- Mejor fitness obtenido: **453.91** (tasa mutación 0.01)
- Municipios atendidos: **125 de 267**
- Presupuesto utilizado: **$10,000,000 MXN**
