# MLOps Titanic Project

Этот проект направлен на демонстрацию навыков MLOps с использованием классического датасета Titanic.
Основной акцент сделан на настройку CI/CD пайплайнов, контейнеризацию, управление зависимостями и версионирование данных и моделей.

### Git Flow

- Работа ведется в отдельных ветках, созданных от main.
- Каждая новая фича оформляется в новой ветке (feature/название-фичи-через-тире).
- После завершения работы создается Pull Request в main
- Итогом: после прохождения всех CI-проверок и ревью Pull Request (даже если вы одни в команде) мержится в main.

> практика Pull Request сохраняется для прозрачности истории и проверки пайплайнов.

### CI/CD

##### Шаг 2: Публикация Docker-образа и Python-пакета

Job `publish-docker`:
- Авторизация в GitHub Container Registry (`ghcr.io`)
- Публикация образа: `docker push`
- Сборка Python-пакета: `pdm build`
- Загрузка артефакта: `.tar.gz` и `.whl` из `dist/`

> Прямая публикация Python-пакета в GitHub Packages невозможна без токена — вместо этого используется `upload-artifact`.

### Как собрать и запустить вручную

```bash
docker build -t mlops-titanic .
```

### Запуск

```bash
docker run --rm mlops-titanic
```

### Data Versioning with DVC + DAGsHub

В этом проекте используется [DVC](https://dvc.org/) для отслеживания данных и моделей. DVC позволяет:

- добавлять данные в репозиторий без хранения их в Git;
- отслеживать изменения в датасетах;
- переключаться между версиями;
- синхронизировать данные с удалённым хранилищем (в нашем случае — [DAGsHub](https://dagshub.com)).

#### Используемые команды

```bash
# Инициализация DVC
dvc init

# Добавление сырых данных
dvc add data/raw/

# Добавление .dvc-файлов и игнорирования в Git
git add data/raw/*.dvc .gitignore

# Настройка удалённого хранилища (DAGsHub)
dvc remote add -d origin-dags https://dagshub.com/katimanova/mlops_titanic.dvc
dvc remote modify origin-dags --local auth basic
dvc remote modify origin-dags --local user katimanova
dvc remote modify origin-dags --local password <DAGsHub токен>

# Загрузка данных в удалённое хранилище
dvc push

# Восстановление данных из удалённого хранилища
dvc pull

# Переключение между версиями данных
git checkout <branch-or-commit>
dvc checkout
```
