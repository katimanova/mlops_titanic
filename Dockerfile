FROM python:3.10-slim

# Устанавливаем pdm
RUN pip install pdm

# Устанавливаем зависимости системы (иначе Snakemake не работает)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    git \
    curl \
    && apt-get clean

WORKDIR /app

COPY pyproject.toml pdm.lock ./
COPY README.md ./

# Устанавливаем зависимости в .venv
RUN pdm install

# Активируем venv вручную
ENV PDM_IGNORE_SAVED_PYTHON=1
ENV PATH="/app/.venv/bin:$PATH"

COPY ./src/ ./src/
COPY ./Snakefile ./Snakefile

# Команда по умолчанию — запуск Snakemake
CMD ["pdm", "run", "snakemake", "--cores", "1", "-p"]