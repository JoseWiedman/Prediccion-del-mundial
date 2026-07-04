import pandas as pd

# ---------------------------------------------------------
# 1. LEER EL DATASET LIMPIO
# ---------------------------------------------------------

df = pd.read_csv("data/results_limpio.csv")

# Convertir la columna date nuevamente a tipo fecha.
# Cuando un CSV se guarda, las fechas vuelven a comportarse como texto.
df["date"] = pd.to_datetime(df["date"])

# Ordenar los partidos por fecha.
# Esto es indispensable porque necesitamos analizar primero
# los partidos antiguos y luego los más recientes.
df = df.sort_values("date").reset_index(drop=True)

print("\nDataset limpio cargado correctamente.")
print(df.head())


# ---------------------------------------------------------
# 2. FUNCIÓN PARA CALCULAR ESTADÍSTICAS RECIENTES
# ---------------------------------------------------------

def calcular_estadisticas_recientes(dataframe, equipo, fecha_partido, n=5):
    """
    Calcula estadísticas de un equipo usando únicamente
    los últimos n partidos jugados antes de una fecha determinada.

    Parámetros:
    ----------
    dataframe : DataFrame
        Dataset completo de partidos.
    equipo : str
        Nombre de la selección que se desea analizar.
    fecha_partido : datetime
        Fecha del partido que queremos predecir.
    n : int
        Cantidad de partidos recientes a revisar.

    Retorna:
    -------
    dict
        goals_for : promedio de goles marcados.
        goals_against : promedio de goles recibidos.
        form_points : puntos obtenidos en los últimos partidos.
        matches_found : cantidad real de partidos encontrados.
    """

    # Buscar únicamente partidos anteriores a la fecha actual
    partidos_previos = dataframe[
        (
            (dataframe["home_team"] == equipo)
            | (dataframe["away_team"] == equipo)
        )
        & (dataframe["date"] < fecha_partido)
    ]

    # Ordenar de más reciente a más antiguo
    partidos_previos = partidos_previos.sort_values(
        "date",
        ascending=False
    )

    # Tomar solo los últimos n partidos
    partidos_previos = partidos_previos.head(n)

    # Si el equipo no tiene partidos históricos previos
    if len(partidos_previos) == 0:
        return {
            "goals_for": 0,
            "goals_against": 0,
            "form_points": 0,
            "matches_found": 0
        }

    goles_favor = []
    goles_contra = []
    puntos = []

    # Recorrer partido por partido
    for _, partido in partidos_previos.iterrows():

        # Caso 1: el equipo jugó como local
        if partido["home_team"] == equipo:

            goles_favor.append(partido["home_score"])
            goles_contra.append(partido["away_score"])

            if partido["home_score"] > partido["away_score"]:
                puntos.append(3)
            elif partido["home_score"] == partido["away_score"]:
                puntos.append(1)
            else:
                puntos.append(0)

        # Caso 2: el equipo jugó como visitante
        else:

            goles_favor.append(partido["away_score"])
            goles_contra.append(partido["home_score"])

            if partido["away_score"] > partido["home_score"]:
                puntos.append(3)
            elif partido["away_score"] == partido["home_score"]:
                puntos.append(1)
            else:
                puntos.append(0)

    # Retornar estadísticas resumidas
    return {
        "goals_for": sum(goles_favor) / len(goles_favor),
        "goals_against": sum(goles_contra) / len(goles_contra),
        "form_points": sum(puntos),
        "matches_found": len(partidos_previos)
    }


# ---------------------------------------------------------
# 3. PROBAR LA FUNCIÓN CON UN EJEMPLO
# ---------------------------------------------------------

ejemplo_equipo = "Colombia"
ejemplo_fecha = pd.Timestamp("2025-01-01")

estadisticas_colombia = calcular_estadisticas_recientes(
    dataframe=df,
    equipo=ejemplo_equipo,
    fecha_partido=ejemplo_fecha,
    n=5
)

print("\nEjemplo de estadísticas recientes:")
print("Equipo:", ejemplo_equipo)
print("Fecha de referencia:", ejemplo_fecha)
print(estadisticas_colombia)


# ---------------------------------------------------------
# 4. CREAR LAS FEATURES PARA CADA PARTIDO
# ---------------------------------------------------------

filas_modelo = []

for _, partido in df.iterrows():

    equipo_local = partido["home_team"]
    equipo_visitante = partido["away_team"]
    fecha_partido = partido["date"]

    # Estadísticas del equipo local antes del partido
    estadisticas_local = calcular_estadisticas_recientes(
        dataframe=df,
        equipo=equipo_local,
        fecha_partido=fecha_partido,
        n=5
    )

    # Estadísticas del equipo visitante antes del partido
    estadisticas_visitante = calcular_estadisticas_recientes(
        dataframe=df,
        equipo=equipo_visitante,
        fecha_partido=fecha_partido,
        n=5
    )

    # Crear una fila enriquecida
    filas_modelo.append({

        # Información original del partido
        "date": partido["date"],
        "year": partido["year"],
        "home_team": equipo_local,
        "away_team": equipo_visitante,
        "home_score": partido["home_score"],
        "away_score": partido["away_score"],
        "result": partido["result"],
        "tournament": partido["tournament"],
        "neutral": partido["neutral"],

        # Variables recientes del equipo local
        "home_recent_goals_for": estadisticas_local["goals_for"],
        "home_recent_goals_against": estadisticas_local["goals_against"],
        "home_recent_form": estadisticas_local["form_points"],
        "home_recent_matches": estadisticas_local["matches_found"],

        # Variables recientes del equipo visitante
        "away_recent_goals_for": estadisticas_visitante["goals_for"],
        "away_recent_goals_against": estadisticas_visitante["goals_against"],
        "away_recent_form": estadisticas_visitante["form_points"],
        "away_recent_matches": estadisticas_visitante["matches_found"]
    })


# ---------------------------------------------------------
# 5. CREAR EL NUEVO DATAFRAME
# ---------------------------------------------------------

df_features = pd.DataFrame(filas_modelo)

print("\nPrimeras filas del dataset con features:")
print(df_features.head())

print("\nColumnas creadas:")
print(df_features.columns)

print("\nTamaño del dataset con features:")
print(df_features.shape)


# ---------------------------------------------------------
# 6. ELIMINAR PARTIDOS SIN HISTORIAL SUFICIENTE
# ---------------------------------------------------------

# Para evitar entrenar con valores donde ambos equipos no tienen
# información previa suficiente, conservaremos partidos donde
# cada selección tenga por lo menos 3 partidos históricos previos.

df_features_filtrado = df_features[
    (df_features["home_recent_matches"] >= 3)
    & (df_features["away_recent_matches"] >= 3)
].copy()

print("\nTamaño antes de filtrar:")
print(df_features.shape)

print("\nTamaño después de filtrar:")
print(df_features_filtrado.shape)


# ---------------------------------------------------------
# 7. GUARDAR EL DATASET ENRIQUECIDO
# ---------------------------------------------------------

df_features_filtrado.to_csv(
    "data/results_features.csv",
    index=False
)

print("\nArchivo generado correctamente:")
print("data/results_features.csv")