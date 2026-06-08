FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libmujoco-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

CMD ["python", "src/mujoco_embodied_ai/main.py", "--run_all_enterprise_pipelines"]
