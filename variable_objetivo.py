import pandas as pd

# 1. Leer el dataset original
df = pd.read_csv("data/results.csv")

# 2. Crear una función para saber quién ganó
def obtener_resultado(row):
    if row["home_score"] > row["away_score"]:
        return "H"  # Gana el equipo local
    elif row["home_score"] < row["away_score"]:
        return "A"  # Gana el equipo visitante
    else:
        return "D"  # Empate

# 3. Aplicar la función a cada fila del dataset
df["result"] = df.apply(obtener_resultado, axis=1)

# 4. Mostrar ejemplo en pantalla
print("\nEjemplo con la nueva columna result:")
print(df[["home_team", "away_team", "home_score", "away_score", "result"]].head(10))

# 5. Guardar un nuevo archivo para el siguiente paso
df.to_csv("data/results_con_resultado.csv", index=False)

print("\nArchivo generado correctamente:")
print("data/results_con_resultado.csv")