# Detector de caras

Detecta caras tipo smiley y las clasifica como felices, tristes o neutras. Está pensado para caras con contorno negro sobre fondo blanco.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Comandos básicos

```bash
# Procesar una imagen
python -m smiley ejemplos/tres_caras.png

# Procesar todas las imágenes de una carpeta
python -m smiley mis_fotos/

# Usar la cámara
python -m smiley --camera 0
```

Las imágenes procesadas se guardan en `detectadas/`. Para no guardarlas:

```bash
python -m smiley mis_fotos/ --no-save
```

## Cómo funciona

1. Convierte la imagen a blanco y negro para buscar las partes oscuras.
2. Encuentra los aros y busca los ojos y la boca dentro de cada cara.
3. Usa la forma de la boca para decidir si la cara está feliz, triste o neutra.

Los ojos permiten clasificar la cara aunque esté girada.
