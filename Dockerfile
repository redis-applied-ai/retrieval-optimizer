# Use the official Python 3.11 image from the Docker Hub
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy the deps file into the container
COPY ./poetry.lock ./pyproject.toml ./

# Install the dependencies before copying code for faster rebuild
RUN poetry install --all-extras --no-interaction

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI server for the labeling app
CMD ["poetry", "run", "uvicorn", "label_app.main:app", "--host", "0.0.0.0", "--port", "8000"]