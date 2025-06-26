import os
import json
from typing import List
from transformers import AutoTokenizer

# Parametros de entrada
RUTA_ENTRADA = "data/TinyStoriesV2-GPT4-train.txt"
RUTA_SALIDA = "data/train.jsonl"
TAMANIO_CHUNK = 128
MAX_LINEAS = 10000  # Cambia si quieres menos

# Tokenizador
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def cargar_texto(ruta_archivo: str, max_lineas: int = None) -> List[str]:
    """
    Lee un archivo de texto línea por línea y elimina líneas vacías.
    """
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        if max_lineas:
            lineas = lineas[:max_lineas]
        return [linea.strip() for linea in lineas if linea.strip()]

def limpiar_texto(lineas: List[str]) -> List[str]:
    """
    Limpia caracteres especiales y espacios múltiples de cada línea.
    """
    return [' '.join(linea.replace('\n', ' ').split()) for linea in lineas]

def dividir_en_chunks(lineas: List[str], tamanio: int) -> List[str]:
    """
    Une el texto completo, tokeniza y divide en fragmentos de tamaño fijo.
    """
    tokens = tokenizer(" ".join(lineas), return_tensors=None)["input_ids"]
    chunks = [tokens[i:i+tamanio] for i in range(0, len(tokens), tamanio)
              if len(tokens[i:i+tamanio]) == tamanio]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def guardar_chunks(chunks: List[str], ruta_salida: str):
    """
    Guarda una lista de fragmentos en un archivo JSONL con campo 'text'.
    """
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            json.dump({"text": chunk}, f)
            f.write('\n')

def main():
    if not os.path.exists(RUTA_ENTRADA):
        print(f"No se encontró el archivo: {RUTA_ENTRADA}")
        return

    lineas = cargar_texto(RUTA_ENTRADA, max_lineas=MAX_LINEAS)
    texto_limpio = limpiar_texto(lineas)
    chunks = dividir_en_chunks(texto_limpio, tamanio=TAMANIO_CHUNK)
    guardar_chunks(chunks, RUTA_SALIDA)
    print(f"Guardado {len(chunks)} fragmentos en {RUTA_SALIDA}")

if __name__ == "__main__":
    main()
