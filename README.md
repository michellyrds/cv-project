# Ex Machina, a IA de detectar faces
![banner](media/banner.png "banner")

Aplicação de visão computacional desenvolvida para o 7º semestre do curso de Sistemas de Informação - EACH USP.  


SO: Ubuntu 20.04.4 LTS
- virtualenv python: [pyenv-installer](https://github.com/pyenv/pyenv-installer) | [pyenv](https://github.com/pyenv/pyenv-virtualenv)
- framework: [Streamlit](https://streamlit.io/)

## Setup local

```terminal
make environment
```

```terminal
make requirements
```

## Docker setup
```terminal
make docker-build
```

### Iniciando a aplicação
```terminal
make start
```
ou via Docker:
```terminal
make docker-start
```

