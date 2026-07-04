import pandas as pd             


#1. Vamos a cargar el dataset "Caracteristicas de resultados.csv"

#Leemos el archivo generado en el paso anterior
#Este archivo contiene datos historicos y las variables
#creadas con la forma reciente de cada seleccion    
df = pd.read_csv("data/results_features.csv")

#Al leer un CSV, las fechas normalmente se cargan como texto 
#Por eso convertimos la columna date nuevamente a formato fecha
df["date"] = pd.to_datetime(df["date"])

print("\n------------------------------------------------")
print("Paso 14 - Preparacion para entrenamiento")
print("---------------------------------------------------")

print("\nPrimeras 5 filas del dataset")
print(df.head())

print("\nTamaño del dataset:")
print(df.shape)

print("\nColumnas disponibles:")
print(df.columns.tolist())

#2. Revisar valores nulos

#Antes de entrenar un modelo, revisamos si existen datos vacios
#Un valor nulo puede causar errores o afectar el entrenamiento
print("\nValores nulos por columna:")
print(df.isnull().sum())

#3. Definir variables de entrada

#Estas variables son la informacion disponible ANTES de que ocurra un partido

columna_features = [

#Indica si el partido se jugó en campo neutral
#False
#True
"neutral",

# Informacion reciente del equipo local
"home_recent_goals_for",
"home_recent_goals_against",
"home_recent_form",
"home_recent_matches",


# Informacion reciente del equipo visitante 
"away_recent_goals_for",
"away_recent_goals_against",
"away_recent_form",
"away_recent_matches",

]

# X Contiene las columnas que el modelo utilizará como entrada.
X = df[columna_features].copy()

print("\nVariables de entrada seleccionadas")
print(X.head())

#4. Definir la variable objetivo    

#y contiene la respuesta que el modelo debe aprender 
#
#H = gana el local
#D = Empate
#A= Gana visitante
y = df["result"].copy()

print("\nVariable objetivo")
print(y.head())

print("\nCantidad de resultados por categoria:")
print(y.value_counts())

#5. Convertir variables a formato numerico

#La columna neutral puede estar como True o False
#Los modelos trabajan mejor con datos numericos
#
#False pasa a ser 0
#True pasa a ser 1
X["neutral"] = X["neutral"].astype(int)

print("\nVista de X despues de convertir neutral a numero:")
print(X.head())

print("\nTipos de datos de X:")
print(X.dtypes)

# ---------------------------------------------------------
# 6. VALIDAR QUE NO HAYA NULOS EN X O y
# ---------------------------------------------------------
 
print("\nValores nulos en X:")
print(X.isnull().sum())
 
print("\nValores nulos en y:")
print(y.isnull().sum())
 
# Si hubiera valores nulos, los eliminamos para evitar errores
# durante el entrenamiento del modelo.
df = df.dropna(subset= columnas_features + ["result"]).copy()
 
print("\nTamaño del dataset después de eliminar nulos:")
print(df.shape)

# ---------------------------------------------------------
# 7. DIVIDIR LOS DATOS POR FECHA
# ---------------------------------------------------------
 
# En este proyecto NO usamos una división aleatoria.
#
# Usamos una división cronológica porque el fútbol cambia con el tiempo.
# El modelo debe aprender con partidos pasados y ser evaluado
# con partidos más recientes.
#
# Entrenamiento: partidos antes de 2020
# Prueba: partidos desde 2020 en adelante
 
train = df[df["year"] < 2020].copy()
test = df[df["year"] >= 2020].copy()
 
print("\n----------------------------------------------")
print("DIVISIÓN CRONOLÓGICA")
print("----------------------------------------------")
 
print("\nCantidad de partidos para entrenamiento:")
print(train.shape)
 
print("\nCantidad de partidos para prueba:")
print(test.shape)
 
print("\nRango de fechas de entrenamiento:")
print(train["date"].min(), "hasta", train["date"].max())
 
print("\nRango de fechas de prueba:")
print(test["date"].min(), "hasta", test["date"].max())

# ---------------------------------------------------------
# 8. CREAR X_train, y_train, X_test Y y_test
# ---------------------------------------------------------
 
# X_train contiene las variables de entrada de los partidos antiguos.
X_train = train[columnas_features].copy()
 
# y_train contiene los resultados reales de los partidos antiguos.
y_train = train["result"].copy()
 
# X_test contiene las variables de entrada de los partidos recientes.
X_test = test[columnas_features].copy()
 
# y_test contiene los resultados reales de los partidos recientes.
y_test = test["result"].copy()
 
# Convertimos neutral a número en ambos conjuntos.
X_train["neutral"] = X_train["neutral"].astype(int)
X_test["neutral"] = X_test["neutral"].astype(int)

# ---------------------------------------------------------
# 9. REVISAR LOS DATOS FINALES
# ---------------------------------------------------------
 
print("\n----------------------------------------------")
print("RESUMEN FINAL PARA MACHINE LEARNING")
print("----------------------------------------------")
 
print("\nTamaño de X_train:")
print(X_train.shape)
 
print("\nTamaño de y_train:")
print(y_train.shape)
 
print("\nTamaño de X_test:")
print(X_test.shape)
 
print("\nTamaño de y_test:")
print(y_test.shape)
 
print("\nPrimeras filas de X_train:")
print(X_train.head())
 
print("\nPrimeros valores de y_train:")
print(y_train.head())
 
print("\nDistribución de resultados en entrenamiento:")
print(y_train.value_counts())
 
print("\nDistribución de resultados en prueba:")
print(y_test.value_counts())

# ---------------------------------------------------------
# 10. GUARDAR LOS ARCHIVOS PREPARADOS
# ---------------------------------------------------------
 
# Guardamos los cuatro archivos que serán utilizados
# en el siguiente paso para entrenar el modelo.
 
X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
 
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
 
print("\n----------------------------------------------")
print("ARCHIVOS GENERADOS CORRECTAMENTE")
print("----------------------------------------------")
 
print("data/X_train.csv")
print("data/y_train.csv")
print("data/X_test.csv")
print("data/y_test.csv")