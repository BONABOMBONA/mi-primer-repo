# Perfilador de Datasets

Herramienta que analiza archivos CSV y genera reportes de calidad de datos.

## Requisitos

- Python 3.8 o superior

## Configuración inicial

```bash
cd ~/Documentos/2026-2/mi-primer-repo/Reto_5
mkdir -p data outputs

cat > data/ventas.csv << 'DATOS'
fecha,producto,cantidad,precio,vendedor
2026-01-01,Laptop,2,15000.00,Ana
2026-01-02,Mouse,10,250.00,Bob
2026-01-03,Teclado,,800.00,Ana
2026-01-04,Monitor,3,,Carlos
2026-01-05,Laptop,1,15000.00,
DATOS
```

## Uso

### Sin argumentos (usa archivos por defecto)
```bash
python3 main.py
```
Lee `data/ventas.csv` y guarda el perfil en `outputs/perfil_ventas.csv`.

### Con argumentos (cualquier CSV)
```bash
python3 main.py --input data/archivo.csv --output outputs/perfil.csv
```

## Autor

Dana Paola Soria López — IPN, Semestre Febrero-Julio 2026
