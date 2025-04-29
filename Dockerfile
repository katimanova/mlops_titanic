FROM python:3.10-slim

RUN pip install pdm

WORKDIR /app

COPY pyproject.toml pdm.lock ./

RUN pdm install --no-self --prod

COPY . .

CMD ["pdm", "run", "python", "src/mlops_titanic/train.py"]
