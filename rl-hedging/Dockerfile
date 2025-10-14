# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependency definition files
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
# --no-root: don't install the project itself, only dependencies
# This is a good practice for docker layer caching
RUN poetry install --no-root --no-dev

# Copy the rest of the application code
COPY . .

# Install the project itself in editable mode
RUN poetry install --no-dev

# Set the entrypoint
ENTRYPOINT ["poetry", "run", "python"]