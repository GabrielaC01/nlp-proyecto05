"""
Este script limpia un corpus de texto plano y lo divide en fragmentos de tamaño fijo.
Los fragmentos se guardan en un archivo JSONL, que luego será usado para entrenar
el modelo de lenguaje.

Parámetros ajustables:
- TAMANIO_CHUNK controla el balance entre cobertura del contexto y consumo de memoria.
  - Chunk grande: más contexto, más memoria
  - Chunk pequeño: menos contexto, menos memoria, más variedad de ejemplos
"""


import os
import json
from typing import List
from transformers import AutoTokenizer
import re

# Parametros de entrada
RUTA_ENTRADA = "data/corpus.txt"
RUTA_SALIDA = "data/train.jsonl"
TAMANIO_CHUNK = 512
MAX_LINEAS = 10000 

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def cargar_texto(ruta_archivo: str, max_lineas: int) -> List[str]:
    """
    Lee un archivo de texto línea por línea y elimina líneas vacías
    """
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        if max_lineas:
            lineas = lineas[:max_lineas]
        return [linea.strip() for linea in lineas if linea.strip()]

def limpiar_texto(lineas: List[str]) -> List[str]:
    """
    Elimina saltos de línea, convierte espacios múltiples en uno solo y filtra caracteres no ASCII
    """
    texto_limpio = []
    for linea in lineas:
        # Elimina saltos de línea y espacios múltiples
        linea = ' '.join(linea.replace('\n', ' ').split())
        # Filtra caracteres no ASCII
        linea = re.sub(r'[^\x00-\x7F]+', '', linea)
        if linea:
            texto_limpio.append(linea)
    return texto_limpio

def dividir_en_chunks(lineas: List[str], tamanio: int) -> List[str]:
    """
    Une el texto completo, tokeniza y divide en fragmentos de tamaño fijo
    """
    tokens = tokenizer(" ".join(lineas), return_tensors=None)["input_ids"]
    chunks = [tokens[i:i+tamanio] for i in range(0, len(tokens), tamanio)
              if len(tokens[i:i+tamanio]) == tamanio]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def guardar_chunks(chunks: List[str], ruta_salida: str):
    """
    Guarda una lista de fragmentos en un archivo JSONL con campo 'text'
    """
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            json.dump({"text": chunk}, f)
            f.write('\n')

def main():
    if not os.path.exists(RUTA_ENTRADA):
        print(f"No se encontró el archivo: {RUTA_ENTRADA}")
        return

    lineas = cargar_texto(RUTA_ENTRADA, MAX_LINEAS)
    texto_limpio = limpiar_texto(lineas)
    chunks = dividir_en_chunks(texto_limpio, TAMANIO_CHUNK)
    guardar_chunks(chunks, RUTA_SALIDA)
    print(f"Guardado {len(chunks)} fragmentos en {RUTA_SALIDA}")

if __name__ == "__main__":
    main()
