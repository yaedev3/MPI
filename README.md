# Como usar GitHub

## Instalación de git en Linux

`sudo apt-get install git-core`

## Comandos básicos

### Para clonar (descargar) este repositorio repositorio

`git clone https://github.com/yaedev3/MPI.git`  

### Para revisar el estado del repositorio

`git status`

### Para añadir los archivos modificados

`git add .`

### Para guardar un cambio

`git commit -m "comentario del cambio"`

### Para subir uno o varios cambios

`git push`

### Para eliminar el ultimo cambio sin guardar

`git stash`

## Ejemplo

#### Verifico el estado del repositorio antes de agregar archivos

`git status`

#### Agrego los archivos que modifqué

`git add .` 

#### Verifico el estado del repositorio despues de agregar archivos

`git status`

#### Guardo el cambio

`git commit -m "Se modifico el archivo x"`

#### Subo el cambio

`git push`