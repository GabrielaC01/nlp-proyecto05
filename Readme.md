<h2 align="center">
<p>Proyecto 5: Preentrenamiento ligero de un LLM </p>
</h2>

## Descripción
Este proyecto implementa el preentrenamiento ligero de un modelo de lenguaje (LLM) utilizando un Transformer de tamaño reducido. Incluye procesamiento de corpus, enmascaramiento dinámico (masking por token y por fragmento), y evaluación de métricas como perplexity y diversidad léxica. El objetivo es entender el pipeline completo de entrenamiento de un modelo de lenguaje en entornos con recursos limitados

## Objetivos

- Limpiar un corpus de texto y dividirlo en fragmentos de contexto fijo
- Implementar un collator que aplique enmascaramiento dinámico por token y estático por fragmento
- Definir un modelo Transformer pequeño y entrenarlo
- Registrar métricas como perplexity y distinct-1/distinct-2 durante el entrenamiento
- Generar texto corto a partir del modelo y analizar su coherencia

## Dataset
Se utilizó el corpus TinyStories, disponible en Hugging Face: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main), que contiene cuentos cortos generados sintéticamente por modelos GPT-3.5 y GPT-4.

En este proyecto se trabajó con la versión inicial TinyStories-train.txt, que incluye textos generados por ambos modelos

