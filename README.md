# Requisito 2: Preprocesamiento y Vectorización del Corpus

## 📋 Descripción
Este requisito procesa un corpus de 680 discursos históricos, genera embeddings semánticos de 384 dimensiones usando el modelo `all-MiniLM-L6-v2` y los almacena en MongoDB configurado con Replica Set.

## 🎯 Objetivos
1. Procesar archivos de texto del corpus
2. Generar identificadores únicos SHA-256
3. Crear embeddings semánticos de cada discurso
4. Almacenar en MongoDB con replicación
5. Validar la integridad de los datos

## 🚀 Instalación Rápida

### Opción 1: Script Automatizado
```bash
# Dar permisos de ejecución
chmod +x setup_requisito2.sh

# Ejecutar el script de configuración
./setup_requisito2.sh
```

### Opción 2: Instalación Manual

#### 1. Verificar Requisito 1
```bash
# Verificar que los contenedores estén corriendo
docker ps | grep mongo

# Deberías ver:
# mongo1 ... 0.0.0.0:3001->27017/tcp
# mongo2 ... 0.0.0.0:3002->27017/tcp
# mongo3 ... 0.0.0.0:3003->27017/tcp
```

#### 2. Crear entorno virtual
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # En macOS/Linux
# o
venv\Scripts\activate     # En Windows
```

#### 3. Instalar dependencias
```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
```

#### 4. Preparar el corpus
```bash
# Verificar archivos
ls DiscursosOriginales/*.txt | wc -l
# Debería mostrar: 680
```

## 📂 Estructura de Archivos

```
laboratorio2/
├── docker-compose.yml          
├── DiscursosOriginales/        
│   ├── 91265.txt
│   ├── 83107.txt
│   └── ...
├── requisito2_procesamiento.py # Script principal
├── requirements.txt            # Dependencias Python
└── venv/                       # Entorno virtual Python
```

## 🔧 Ejecución del Requisito 2

### 1. Ejecutar el procesamiento
```bash
python3 requisito2_procesamiento.py
```

### 2. Output esperado
```
🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟
LABORATORIO 2 - REQUISITO 2
Preprocesamiento y Vectorización del Corpus
🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟

🤖 Cargando modelo de embeddings...
✅ Modelo cargado exitosamente
   - Dimensión de embeddings: 384

🔌 Conectando a MongoDB Replica Set...
✅ Conexión exitosa a MongoDB (puerto 3001)

📊 Información del servidor MongoDB:
   - Versión: 8.0.3
   - Replica Set: rs
   - Miembros activos: 3
     • mongo1:27017: PRIMARY
     • mongo2:27017: SECONDARY
     • mongo3:27017: SECONDARY

============================================================
🚀 INICIANDO PROCESAMIENTO DEL CORPUS
============================================================
📁 Encontrados 680 archivos para procesar
📍 Carpeta: /path/to/DiscursosOriginales

Procesando discursos: 100%|████████| 680/680 [00:45<00:00, 15.11archivo/s]

============================================================
📊 RESUMEN DEL PROCESAMIENTO
============================================================
✅ Documentos procesados exitosamente: 679
⚠️  Documentos duplicados omitidos: 1
❌ Errores durante el procesamiento: 0
⏱️  Tiempo total de procesamiento: 45.00 segundos
⚡ Velocidad promedio: 15.09 docs/segundo
📦 Total documentos en la colección: 679
============================================================
```

### 3. Verificar resultados
```bash
python3 verificar_requisito2.py
```

## 📊 Estructura del Documento MongoDB

Cada documento procesado tiene exactamente esta estructura:

```json
{
    "_id": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
    "texto": "Contenido completo del discurso...",
    "embedding": [0.0234, -0.1245, 0.5678, ..., 0.0432]
}
```

- **_id**: Hash SHA-256 del texto completo (64 caracteres hexadecimales)
- **texto**: Contenido íntegro del discurso
- **embedding**: Vector de 384 dimensiones flotantes

## 🔍 Verificación en MongoDB

### Conectarse a MongoDB
```bash
docker exec -it mongo1 mongosh
```
### Comandos de verificación
```javascript
// Usar la base de datos
use Política

// Contar documentos
db.Discursos.countDocuments()
// Resultado esperado: 679

// Ver estructura de un documento
db.Discursos.findOne({}, {texto: 0, embedding: {$slice: 5}})

// Verificar dimensión de embeddings
db.Discursos.aggregate([
    {$project: {embedding_size: {$size: "$embedding"}}},
    {$limit: 1}
])
// Resultado esperado: embedding_size: 384
exit
```

## 🐛 Solución de Problemas

### Error: "No se encuentra la carpeta DiscursosOriginales"
```bash
# Verificar si existe el archivo comprimido
ls DiscursosOriginales.rar

# Descomprimir
unrar x DiscursosOriginales.rar
```

### Error: "Connection refused" o "No se pudo conectar a MongoDB"
```bash
# Verificar contenedores
docker ps

# Si no están corriendo, levantarlos
cd ../Requisito1  # o donde esté docker-compose.yml
docker-compose up -d
```

### Error: "ModuleNotFoundError"
```bash
# Verificar que el entorno virtual esté activado
which python3
# Debe mostrar: .../venv/bin/python3

# Reinstalar dependencias
pip install -r requirements.txt
```

## 📈 Métricas de Rendimiento

| Métrica | Valor Típico | Notas |
|---------|--------------|-------|
| Documentos procesados | 679/680 | 1 duplicado es normal |
| Tiempo total | 30-60 seg | Depende del hardware |
| Velocidad | 10-20 docs/seg | Sin GPU |
| Tamaño embedding | 384 dims | Fijo por el modelo |
| Uso de RAM | ~2-3 GB | Durante procesamiento |

## ✅ Criterios de Éxito

- [x] 679 documentos procesados (99.85%)
- [x] 0 errores críticos
- [x] Todos los embeddings de 384 dimensiones
- [x] Documentos replicados en 3 nodos
- [x] Validación exitosa

## 📝 Logs y Archivos Generados

1. **procesamiento_requisito2.log**: Log detallado del procesamiento
2. **Base de datos MongoDB**: Colección "Discursos" en base "Política"

## 🔗 Integración con otros Requisitos

- **Requisito 1**: Debe estar funcionando (MongoDB Replica Set)
- **Requisito 3**: Usará estos embeddings para búsqueda semántica
- **Requisito 4**: Probará alta disponibilidad con estos datos
---

*Este requisito es parte del Laboratorio 2 de MongoDB y debe ejecutarse después del Requisito 1 (configuración del Replica Set).*