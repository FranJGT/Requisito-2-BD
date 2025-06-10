#!/usr/bin/env python3
"""
Requisito 2: Preprocesamiento del Corpus
Vectorización e Inserción en MongoDB

Este script procesa discursos históricos, genera embeddings semánticos
y los almacena en MongoDB configurado con Replica Set.
"""

import os
import sys
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, WriteConcern, errors
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('procesamiento_requisito2.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CorpusProcessor:
    """
    Procesador de corpus de discursos para vectorización e inserción en MongoDB.
    Diseñado para trabajar con el Replica Set configurado en el Requisito 1.
    """
    
    def __init__(self, corpus_path: str):
        """
        Inicializa el procesador con la ruta del corpus.
        
        Args:
            corpus_path: Ruta a la carpeta con los archivos .txt
        """
        self.corpus_path = Path(corpus_path)
        self.client = None
        self.db = None
        self.collection = None
        self.model = None
        
        self.stats = {
            'procesados': 0,
            'errores': 0,
            'duplicados': 0,
            'tiempo_inicio': datetime.now(),
            'archivos_error': []
        }
        
        self.MONGO_PORTS = [3001, 3002, 3003]
        self.REPLICA_SET = "rs"
        self.DATABASE_NAME = "Política"  
        self.COLLECTION_NAME = "Discursos"
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Inicializa modelo de embeddings y conexión a MongoDB"""
        logger.info("🤖 Cargando modelo de embeddings...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Modelo cargado exitosamente")
            
            logger.info(f"   - Dimensión de embeddings: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
        
        self._connect_to_mongodb()
    
    def _connect_to_mongodb(self):
        """
        Establece conexión con MongoDB Replica Set.
        Intenta múltiples estrategias de conexión.
        """
        logger.info("🔌 Conectando a MongoDB Replica Set...")
        
        for port in self.MONGO_PORTS:
            try:
                logger.info(f"   Intentando puerto {port}...")
                self.client = MongoClient(
                    f'mongodb://localhost:{port}',
                    serverSelectionTimeoutMS=5000,
                    directConnection=True
                )
                self.client.admin.command('ping')
                logger.info(f"✅ Conexión exitosa a MongoDB (puerto {port})")
                break
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"   ⚠️  No se pudo conectar al puerto {port}")
                continue
        
        if not self.client:
            try:
                logger.info("   Intentando conexión a Replica Set completo...")
                replica_set_uri = f"mongodb://localhost:{self.MONGO_PORTS[0]},localhost:{self.MONGO_PORTS[1]},localhost:{self.MONGO_PORTS[2]}/?replicaSet={self.REPLICA_SET}"
                
                self.client = MongoClient(
                    replica_set_uri,
                    serverSelectionTimeoutMS=10000
                )
                self.client.admin.command('ping')
                logger.info("✅ Conexión exitosa a MongoDB Replica Set")
                
            except Exception as e:
                logger.error(f"❌ No se pudo conectar a MongoDB: {e}")
                raise ConnectionError("No se pudo establecer conexión con MongoDB. Verifica que el Requisito 1 esté funcionando.")
            
        self.db = self.client[self.DATABASE_NAME]
        self.collection = self.db[self.COLLECTION_NAME]
        
        self._show_connection_info()
    
    def _show_connection_info(self):
        """Muestra información sobre la conexión establecida"""
        try:
            server_info = self.client.server_info()
            logger.info(f"📊 Información del servidor MongoDB:")
            logger.info(f"   - Versión: {server_info.get('version', 'desconocida')}")
            
            if self.client.admin.command('isMaster').get('setName'):
                status = self.client.admin.command('replSetGetStatus')
                logger.info(f"   - Replica Set: {status['set']}")
                logger.info(f"   - Miembros activos: {len(status['members'])}")
                
                for member in status['members']:
                    state = member['stateStr']
                    name = member['name']
                    logger.info(f"     • {name}: {state}")
            
        except Exception as e:
            logger.warning(f"⚠️  No se pudo obtener información completa del servidor: {e}")
    
    def generate_sha256(self, text: str) -> str:
        """
        Genera hash SHA-256 del texto.
        
        Args:
            text: Texto a hashear
            
        Returns:
            Hash hexadecimal de 64 caracteres
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera embedding del texto usando sentence-transformers.
        
        Args:
            text: Texto a vectorizar
            
        Returns:
            Lista de floats representando el embedding
        """
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            raise
    
    def process_file(self, filepath: Path) -> Optional[Dict]:
        """
        Procesa un archivo individual y retorna el documento MongoDB.
        
        Args:
            filepath: Ruta al archivo .txt
            
        Returns:
            Diccionario con la estructura del documento o None si hay error
        """
        try:
            texto = None
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        texto = f.read().strip()
                    break
                except UnicodeDecodeError:
                    continue
            
            if texto is None:
                raise ValueError(f"No se pudo leer el archivo con ningún encoding")
            
            if not texto:
                raise ValueError("Archivo vacío")
            
            doc_id = self.generate_sha256(texto)
            
            embedding = self.generate_embedding(texto)
            
            documento = {
                "_id": doc_id,
                "texto": texto,
                "embedding": embedding
            }
            
            return documento
            
        except Exception as e:
            logger.error(f"Error procesando {filepath.name}: {str(e)}")
            self.stats['archivos_error'].append({
                'archivo': filepath.name,
                'error': str(e)
            })
            return None
    
    def process_corpus(self):
        """
        Procesa todos los archivos del corpus.
        Método principal que coordina todo el procesamiento.
        """
        logger.info("\n" + "="*60)
        logger.info("🚀 INICIANDO PROCESAMIENTO DEL CORPUS")
        logger.info("="*60)
        
        txt_files = list(self.corpus_path.glob('*.txt'))
        total_files = len(txt_files)
        
        if total_files == 0:
            logger.error(f"❌ No se encontraron archivos .txt en {self.corpus_path}")
            return
        
        logger.info(f"📁 Encontrados {total_files} archivos para procesar")
        logger.info(f"📍 Carpeta: {self.corpus_path.absolute()}")
        
        with tqdm(total=total_files, desc="Procesando discursos", unit="archivo") as pbar:
            for idx, filepath in enumerate(txt_files):
                try:
                    documento = self.process_file(filepath)
                    
                    if documento is None:
                        self.stats['errores'] += 1
                        pbar.update(1)
                        continue
                    
                    result = self.collection.with_options(write_concern=WriteConcern(w="majority", wtimeout=5000)).insert_one(documento)
                    
                    self.stats['procesados'] += 1

                    pbar.set_postfix({
                        'Procesados': self.stats['procesados'],
                        'Errores': self.stats['errores'],
                        'Duplicados': self.stats['duplicados']
                    })
                    
                except errors.DuplicateKeyError:
                    logger.warning(f"⚠️  Documento duplicado: {filepath.name}")
                    self.stats['duplicados'] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error con {filepath.name}: {str(e)}")
                    self.stats['errores'] += 1
                    self.stats['archivos_error'].append({
                        'archivo': filepath.name,
                        'error': str(e)
                    })
                
                pbar.update(1)
                
                if (idx + 1) % 50 == 0:
                    time.sleep(0.1)
        
        self._print_summary()
    
    def _print_summary(self):
        """Imprime resumen detallado del procesamiento"""
        duracion = (datetime.now() - self.stats['tiempo_inicio']).total_seconds()
        
        print("\n" + "="*60)
        print("📊 RESUMEN DEL PROCESAMIENTO")
        print("="*60)
        print(f"✅ Documentos procesados exitosamente: {self.stats['procesados']}")
        print(f"⚠️  Documentos duplicados omitidos: {self.stats['duplicados']}")
        print(f"❌ Errores durante el procesamiento: {self.stats['errores']}")
        print(f"⏱️  Tiempo total de procesamiento: {duracion:.2f} segundos")
        print(f"⚡ Velocidad promedio: {self.stats['procesados']/duracion:.2f} docs/segundo")
        print(f"📦 Total documentos en la colección: {self.collection.count_documents({})}")
        print("="*60)
        
        if self.stats['archivos_error']:
            print("\n⚠️  ARCHIVOS CON ERRORES:")
            print("-"*60)
            for error_info in self.stats['archivos_error'][:5]: 
                print(f"   • {error_info['archivo']}: {error_info['error']}")
            if len(self.stats['archivos_error']) > 5:
                print(f"   ... y {len(self.stats['archivos_error']) - 5} errores más")
    
    def validate_collection(self):
        """
        Valida la integridad de la colección después del procesamiento.
        Verifica estructura, consistencia y replicación.
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 VALIDACIÓN DE LA COLECCIÓN")
        logger.info("="*60)
        
        total_docs = self.collection.count_documents({})
        logger.info(f"📊 Total de documentos en la colección: {total_docs}")
        
        if total_docs == 0:
            logger.warning("⚠️  La colección está vacía")
            return False

        sample = self.collection.find_one()
        if sample:
            logger.info("\n✅ Estructura del documento de muestra:")
            logger.info(f"   - ID (SHA-256): {sample['_id'][:32]}...")
            logger.info(f"   - Longitud del texto: {len(sample['texto'])} caracteres")
            logger.info(f"   - Dimensión del embedding: {len(sample['embedding'])}")
            logger.info(f"   - Primeras palabras: {' '.join(sample['texto'].split()[:10])}...")
        
        logger.info("\n📐 Verificando consistencia de embeddings...")
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
        
        if len(sizes) == 1 and sizes[0]['_id'] == 384:
            logger.info(f"✅ Todos los {sizes[0]['count']} documentos tienen embeddings de 384 dimensiones")
        else:
            logger.warning("⚠️  Inconsistencia en las dimensiones de embeddings:")
            for size in sizes:
                logger.warning(f"   - {size['count']} documentos con {size['_id']} dimensiones")
        
        logger.info("\n🔎 Verificando campos requeridos...")
        docs_sin_texto = self.collection.count_documents({"texto": {"$exists": False}})
        docs_sin_embedding = self.collection.count_documents({"embedding": {"$exists": False}})
        
        if docs_sin_texto == 0 and docs_sin_embedding == 0:
            logger.info("✅ Todos los documentos tienen los campos requeridos")
        else:
            if docs_sin_texto > 0:
                logger.warning(f"⚠️  {docs_sin_texto} documentos sin campo 'texto'")
            if docs_sin_embedding > 0:
                logger.warning(f"⚠️  {docs_sin_embedding} documentos sin campo 'embedding'")
        
        logger.info("\n🔧 Verificando estado del Replica Set...")
        try:
            status = self.client.admin.command('replSetGetStatus')
            logger.info(f"✅ Replica Set '{status['set']}' activo")
            logger.info(f"   - Miembros: {len(status['members'])}")
            
            primary_count = sum(1 for m in status['members'] if m['stateStr'] == 'PRIMARY')
            if primary_count == 1:
                logger.info("   - ✅ Primario activo")
            else:
                logger.warning(f"   - ⚠️  {primary_count} primarios detectados (debería ser 1)")
                
        except Exception as e:
            logger.warning(f"⚠️  No se pudo verificar el estado del Replica Set: {e}")
        
        logger.info("="*60)
        
        return True
    
    def create_indexes(self):
        """
        Crea índices adicionales para optimizar las búsquedas futuras.
        """
        logger.info("\n📇 Creando índices para optimización...")
        
        try:
            self.collection.create_index([("texto", "text")], name="text_index")
            logger.info("✅ Índice de texto creado")
            
            self.collection.create_index("embedding", name="embedding_index", sparse=True)
            logger.info("✅ Índice de embedding creado")
            
        except Exception as e:
            logger.warning(f"⚠️  Error creando índices: {e}")
    
    def close(self):
        """Cierra la conexión a MongoDB"""
        if self.client:
            self.client.close()
            logger.info("🔌 Conexión a MongoDB cerrada")


def main():
    """Función principal del script"""
    print("\n" + "🌟"*30)
    print("LABORATORIO 2 - REQUISITO 2")
    print("Preprocesamiento y Vectorización del Corpus")
    print("🌟"*30 + "\n")
    
    CORPUS_PATH = "./DiscursosOriginales"
    
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Error: No se encuentra la carpeta '{CORPUS_PATH}'")
        print("\n📝 Instrucciones:")
        print("1. Asegúrate de haber descomprimido el archivo DiscursosOriginales.rar")
        print("2. La carpeta debe estar en el mismo directorio que este script")
        print("3. Verifica que la carpeta contenga archivos .txt")
        return 1
    
    try:
        processor = CorpusProcessor(CORPUS_PATH)
        
        processor.process_corpus()
        
        processor.validate_collection()
        
        processor.create_indexes()

        processor.close()
        
        print("\n✅ ¡Procesamiento completado exitosamente!")
        print("📝 Revisa el archivo 'procesamiento_requisito2.log' para más detalles")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Procesamiento interrumpido por el usuario")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        logger.exception("Error no manejado durante el procesamiento")
        return 1


if __name__ == "__main__":
    exit(main())