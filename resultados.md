# Resultados de generación

## Sin prompt:
- Distinct-1: 0.1645
- Distinct-2: 0.7231
- Perplexity: 230.70

## Con prompt 'The little cat':
- Distinct-1: 0.2875
- Distinct-2: 0.7548
- Perplexity: 184.64

## Análisis técnico:

Al generar sin prompt, el modelo parte de un contexto vacío, lo que incrementa la perplexity (**230.70**) debido a la falta de referencia previa. La diversidad léxica es aceptable (**distinct-2: 0.7231**), pero el contenido carece de coherencia temática

Con el prompt "The little cat", el modelo inicia con un contexto claro, reduciendo la perplexity (**184.64**) y aumentando la diversidad (**distinct-2: 0.7548**). Esto indica que el modelo aprovecha mejor la información inicial, generando secuencias más variadas y estructuradas
