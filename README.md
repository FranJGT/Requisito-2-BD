# Requisito 2: Preprocesamiento y Vectorización del Corpus

## 📋 Descripción
Este requisito procesa un corpus de discursos históricos, genera embeddings semánticos y los almacena en MongoDB para búsquedas por similitud.

## 🚀 Guía de Ejecución Paso a Paso

### 1️⃣ Prerrequisitos

Asegúrate de tener instalado:
- Python 3.8 o superior
- Docker Desktop
- MongoDB ejecutándose en contenedores (del Requisito 1)

Verifica que MongoDB esté funcionando:
```bash
docker ps | grep mongo
```

Deberías ver 3 contenedores: mongo1, mongo2, mongo3

### 2️⃣ Preparar el Entorno

#### Clonar o crear el directorio del proyecto:
```bash
mkdir laboratorio2
cd laboratorio2
```

#### Crear entorno virtual (recomendado):
```bash
# En macOS/Linux
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Instalar Dependencias

#### Crear archivo `requirements.txt`:
```bash
cat > requirements.txt << EOF
pymongo==4.6.1
sentence-transformers==2.3.1
numpy==1.24.3
torch>=2.0.0
transformers>=4.36.0
tqdm==4.66.1
EOF
```

#### Instalar las dependencias:
```bash
pip install -r requirements.txt
```

### 4️⃣ Preparar el Corpus de Discursos

#### Descomprimir los archivos de discursos:
```bash
# Si tienes el archivo .rar
unrar x DiscursosOriginales.rar

# O si tienes el archivo .zip
unzip DiscursosOriginales.zip
```

#### Verificar que los archivos estén en la carpeta correcta:
```bash
ls DiscursosOriginales/*.txt | head -5
```

Deberías ver archivos como: `91265.txt`, `83107.txt`, etc.

### 5️⃣ Crear el Script de Procesamiento

#### Crear archivo `requisito2_procesamiento.py`:
```bash
cat > requisito2_procesamiento.py << 'EOF'
import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, WriteConcern, errors
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('procesamiento.log'),
        logging.StreamHandler()
    ]
)

class CorpusProcessor:
    """Procesador de corpus de discursos para vectorización e inserción en MongoDB"""
    
    def __init__(self, corpus_path: str, mongo_uri: str):
        self.corpus_path = Path(corpus_path)
        self.mongo_uri = mongo_uri
        
        # Inicializar modelo de embeddings
        logging.info("Cargando modelo de embeddings...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Conectar a MongoDB
        logging.info("Conectando a MongoDB...")
        self.client = MongoClient(mongo_uri)
        self.db = self.client['Politica']
        self.collection = self.db['Discursos']
        
        # Estadísticas
        self.stats = {
            'procesados': 0,
            'errores': 0,
            'duplicados': 0,
            'tiempo_inicio': datetime.now()
        }
    
    def generate_sha256(self, text: str) -> str:
        """Genera hash SHA-256 del texto"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Genera embedding del texto usando sentence-transformers"""
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logging.error(f"Error generando embedding: {e}")
            raise
    
    def process_file(self, filepath: Path) -> Dict:
        """Procesa un archivo individual y retorna el documento"""
        try:
            # Leer archivo
            with open(filepath, 'r', encoding='utf-8') as f:
                texto = f.read().strip()
            
            if not texto:
                raise ValueError("Archivo vacío")
            
            # Generar hash y embedding
            doc_id = self.generate_sha256(texto)
            embedding = self.generate_embedding(texto)
            
            # Crear documento
            documento = {
                "_id": doc_id,
                "texto": texto,
                "embedding": embedding,
                "metadata": {
                    "archivo_original": filepath.name,
                    "fecha_procesamiento": datetime.now(),
                    "longitud_texto": len(texto),
                    "primeras_palabras": ' '.join(texto.split()[:10]) + "..."
                }
            }
            
            return documento
            
        except Exception as e:
            logging.error(f"Error procesando {filepath.name}: {str(e)}")
            raise
    
    def process_corpus(self):
        """Procesa todos los archivos del corpus"""
        # Obtener archivos .txt
        txt_files = list(self.corpus_path.glob('*.txt'))
        total_files = len(txt_files)
        
        if total_files == 0:
            logging.error(f"No se encontraron archivos .txt en {self.corpus_path}")
            return
        
        logging.info(f"Encontrados {total_files} archivos para procesar")
        
        # Procesar cada archivo
        with tqdm(total=total_files, desc="Procesando discursos") as pbar:
            for filepath in txt_files:
                try:
                    # Procesar archivo
                    documento = self.process_file(filepath)
                    
                    # Insertar en MongoDB
                    self.collection.insert_one(documento)
                    
                    self.stats['procesados'] += 1
                    pbar.set_postfix({'Procesados': self.stats['procesados']})
                    
                except errors.DuplicateKeyError:
                    logging.warning(f"Documento duplicado: {filepath.name}")
                    self.stats['duplicados'] += 1
                    
                except Exception as e:
                    logging.error(f"Error con {filepath.name}: {str(e)}")
                    self.stats['errores'] += 1
                
                pbar.update(1)
        
        self._print_summary()
    
    def _print_summary(self):
        """Imprime resumen del procesamiento"""
        duracion = (datetime.now() - self.stats['tiempo_inicio']).total_seconds()
        
        print("\n" + "="*60)
        print("RESUMEN DEL PROCESAMIENTO")
        print("="*60)
        print(f"✅ Documentos procesados: {self.stats['procesados']}")
        print(f"⚠️  Duplicados omitidos: {self.stats['duplicados']}")
        print(f"❌ Errores: {self.stats['errores']}")
        print(f"⏱️  Tiempo total: {duracion:.2f} segundos")
        print(f"📊 Documentos en BD: {self.collection.count_documents({})}")
        print("="*60)
    
    def validate_collection(self):
        """Valida la integridad de la colección"""
        print("\n🔍 VALIDACIÓN DE LA COLECCIÓN")
        print("="*60)
        
        # Verificar estructura de un documento
        sample = self.collection.find_one()
        if sample:
            print("✅ Estructura del documento:")
            print(f"   - ID (SHA-256): {sample['_id'][:16]}...")
            print(f"   - Longitud texto: {len(sample['texto'])} caracteres")
            print(f"   - Dimensión embedding: {len(sample['embedding'])}")
            print(f"   - Metadata: {list(sample.get('metadata', {}).keys())}")
        
        # Verificar consistencia de embeddings
        pipeline = [
            {"$project": {
                "embedding_size": {"$size": "$embedding"}
            }},
            {"$group": {
                "_id": "$embedding_size",
                "count": {"$sum": 1}
            }}
        ]
        
        sizes = list(self.collection.aggregate(pipeline))
        print(f"\n✅ Consistencia de embeddings:")
        for size in sizes:
            print(f"   - Dimensión {size['_id']}: {size['count']} documentos")
        
        return True

# Script principal
if __name__ == "__main__":
    # Configuración
    CORPUS_PATH = "./DiscursosOriginales"
    MONGO_URI = "mongodb://localhost:27017/?directConnection=true"
    
    # Verificar que la carpeta existe
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Error: No se encuentra la carpeta {CORPUS_PATH}")
        print("Asegúrate de haber descomprimido DiscursosOriginales.rar")
        exit(1)
    
    # Crear procesador
    processor = CorpusProcessor(CORPUS_PATH, MONGO_URI)
    
    # Procesar corpus
    processor.process_corpus()
    
    # Validar resultados
    processor.validate_collection()
EOF
```

### 6️⃣ Ejecutar el Procesamiento

#### Ejecutar el script:
```bash
python3 requisito2_procesamiento.py
```

#### Resultado esperado:
```
2025-06-05 19:15:41,292 - INFO - Cargando modelo de embeddings...
2025-06-05 19:15:44,853 - INFO - Conectando a MongoDB...
2025-06-05 19:15:44,861 - INFO - Encontrados 680 archivos para procesar
Procesando discursos: 100%|████████| 680/680 [00:13<00:00, 50.79it/s]

============================================================
RESUMEN DEL PROCESAMIENTO
============================================================
✅ Documentos procesados: 679
⚠️  Duplicados omitidos: 1
❌ Errores: 0
⏱️  Tiempo total: 13.40 segundos
📊 Documentos en BD: 679
============================================================
```

### 7️⃣ Verificar los Resultados

#### Conectarse a MongoDB y verificar:
```bash
docker exec -it mongo1 mongosh
```

#### Dentro de mongosh:
```javascript
// Cambiar a la base de datos
use Politica

// Contar documentos
db.Discursos.countDocuments()

// Ver un documento de ejemplo
db.Discursos.findOne()

// Verificar que todos tienen embeddings
db.Discursos.find({"embedding": {"$exists": false}}).count()

// Salir
exit
```

### 8️⃣ Verificación con Script Python

#### Crear archivo `verificar_requisito2.py`:
```bash
cat > verificar_requisito2.py << 'EOF'
from pymongo import MongoClient

# Conectar a MongoDB
client = MongoClient('mongodb://localhost:27017/?directConnection=true')
db = client['Politica']
collection = db['Discursos']

# Estadísticas
total = collection.count_documents({})
con_embeddings = collection.count_documents({"embedding": {"$exists": True}})
sin_embeddings = collection.count_documents({"embedding": {"$exists": False}})

# Obtener un documento de muestra
muestra = collection.find_one()

print("📊 VERIFICACIÓN DEL REQUISITO 2")
print("="*50)
print(f"✅ Total de documentos: {total}")
print(f"✅ Documentos con embeddings: {con_embeddings}")
print(f"❌ Documentos sin embeddings: {sin_embeddings}")

if muestra:
    print(f"\n📄 Documento de muestra:")
    print(f"   - ID: {muestra['_id'][:16]}...")
    print(f"   - Longitud del texto: {len(muestra['texto'])} caracteres")
    print(f"   - Dimensión del embedding: {len(muestra['embedding'])}")
    print(f"   - Metadata: {list(muestra.get('metadata', {}).keys())}")

print("\n✅ Requisito 2 completado exitosamente!" if sin_embeddings == 0 else "❌ Hay documentos sin procesar")
EOF
```

#### Ejecutar la verificación:
```bash
python3 verificar_requisito2.py
```

## 🐛 Solución de Problemas

### Error: "No se encuentra la carpeta DiscursosOriginales"
```bash
# Verificar contenido del directorio actual
ls -la

# Si los archivos están en otro lugar, mover o actualizar CORPUS_PATH en el script
```

### Error: "Connection refused" al conectar a MongoDB
```bash
# Verificar que los contenedores estén ejecutándose
docker ps

# Si no están ejecutándose, iniciarlos
docker-compose up -d
```

### Error: "Module not found"
```bash
# Asegurarse de que el entorno virtual esté activado
source venv/bin/activate  # macOS/Linux
# o
venv\Scripts\activate  # Windows

# Reinstalar dependencias
pip install -r requirements.txt
```

### Procesamiento muy lento
```bash
# El modelo se descarga la primera vez (puede tomar varios minutos)
# Las siguientes ejecuciones serán más rápidas
```

## 📝 Logs y Salidas

Los logs se guardan en:
- `procesamiento.log` - Log detallado del procesamiento
- Terminal - Progreso en tiempo real

## ✅ Criterios de Éxito

El requisito se considera completado cuando:
1. ✅ Se procesan al menos 650 documentos (95% del total)
2. ✅ Todos los documentos tienen embeddings de 384 dimensiones
3. ✅ No hay errores críticos durante el procesamiento
4. ✅ La validación final es exitosa

## 🎯 Siguiente Paso

Una vez completado el Requisito 2, puedes proceder con:
- **Requisito 3**: Sistema de búsqueda semántica
- **Requisito 4**: Pruebas de alta disponibilidad

---
