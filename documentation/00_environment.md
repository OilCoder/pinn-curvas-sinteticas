# Entorno de desarrollo Docker

Todo el trabajo de desarrollo ocurre dentro del contenedor Docker. No se instalan dependencias directamente en WSL2.

---

## Prerequisitos

### Docker Engine

Instalar Docker Engine para Ubuntu siguiendo la [guía oficial](https://docs.docker.com/engine/install/ubuntu/). Verificar la instalación:

```bash
docker --version          # Docker Engine 24+
docker compose version    # Docker Compose v2.x
```

### NVIDIA Container Toolkit

Necesario para que el contenedor acceda a la GPU. Ejecutar en WSL2:

```bash
# Agregar el repositorio de NVIDIA
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instalar y configurar
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verificar que la GPU sea visible desde Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Build del contenedor

Desde la raiz del proyecto (`/home/pokinux/pinn-curvas-sinteticas/`):

```bash
docker compose build
```

Esto descarga la imagen base de PyTorch (~5 GB la primera vez) e instala todas las dependencias de `requirements.txt`. Las rebuilds posteriores usan la cache de capas y son mucho mas rapidas.

Para forzar una rebuild limpia (por ejemplo, despues de cambiar `requirements.txt`):

```bash
docker compose build --no-cache
```

---

## Verificar acceso a la GPU

Abrir una sesion interactiva y confirmar que PyTorch detecta la RTX 4080:

```bash
docker compose run --rm dev python -c "
import torch
print('CUDA disponible:', torch.cuda.is_available())
print('Dispositivo:', torch.cuda.get_device_name(0))
print('Memoria total:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"
```

Salida esperada:

```
CUDA disponible: True
Dispositivo: NVIDIA GeForce RTX 4080
Memoria total: 16.0 GB
```

---

## Shell interactivo

Para trabajar de forma interactiva dentro del contenedor:

```bash
docker compose run --rm dev
```

Esto monta el directorio del proyecto en `/workspace` y abre un shell bash. Los cambios en archivos son inmediatos — no hace falta rebuild.

Para salir del contenedor: `exit` o `Ctrl+D`.

---

## Correr los tests

```bash
# Todos los tests con reporte de cobertura
docker compose run --rm dev pytest --cov=src --cov-report=term-missing

# Un modulo especifico
docker compose run --rm dev pytest tests/test_modelo.py -v

# Test rapido sin cobertura
docker compose run --rm dev pytest tests/
```

---

## Correr un script de entrenamiento

```bash
docker compose run --rm dev python src/train.py
```

El flag `--rm` elimina el contenedor al terminar. Si se necesita inspeccionar el estado del contenedor despues de que falle, omitir `--rm`.

---

## Linting y type checking

```bash
# Verificar estilo y errores de codigo
docker compose run --rm dev ruff check src/ tests/

# Aplicar correcciones automaticas
docker compose run --rm dev ruff check --fix src/ tests/

# Verificar tipos estaticamente
docker compose run --rm dev mypy src/
```

---

## Agregar dependencias

1. Agregar el paquete con version pineada a `requirements.txt`:

   ```
   scipy==1.14.0
   ```

2. Rebuild del contenedor para incorporar la nueva dependencia:

   ```bash
   docker compose build
   ```

3. Verificar que la instalacion fue exitosa:

   ```bash
   docker compose run --rm dev python -c "import scipy; print(scipy.__version__)"
   ```

No instalar paquetes con `pip install` directamente en el contenedor en ejecucion — los cambios se pierden al salir. Siempre actualizar `requirements.txt` y hacer rebuild.

---

## Notas sobre permisos en WSL2

El contenedor corre con UID/GID 1000, que tipicamente coincide con el usuario por defecto de WSL2. Si los archivos creados dentro del contenedor aparecen como propiedad de `root` en el host, verificar el UID del usuario WSL2:

```bash
id -u   # debe ser 1000
```

Si el UID es distinto, modificar el `Dockerfile` para que el `useradd` use el UID correcto y hacer rebuild.
