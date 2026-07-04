import pandas as pd

# 1. Leer el archivo generado en el paso anterior
df = pd.read_csv("data/results_con_resultado.csv")

# 2. Convertir la columna date a tipo fecha
df["date"] = pd.to_datetime(df["date"])

# 3. Crear la columna year a partir de la fecha
df["year"] = df["date"].dt.year

# 4. Filtrar partidos desde el año 2000
df_moderno = df[df["year"] >= 2000].copy()

# 5. Mostrar resultados
print("\nFechas convertidas:")
print(df_moderno[["date", "year"]].head())

print("\nDataset desde el año 2000:")
print(df_moderno.shape)

print("\nColumnas disponibles después de la limpieza:")
print(df_moderno.columns)

print("\nEjemplo de datos limpios:")
print(df_moderno.head())

# 6. Guardar archivo limpio para los siguientes pasos
df_moderno.to_csv("data/results_limpio.csv", index=False)

print("\nArchivo generado correctamente:")
print("data/results_limpio.csv")