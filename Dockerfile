FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim

RUN apt-get update && apt-get install -y \
    git curl build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV DEVICE=cpu

WORKDIR /app
COPY . .

RUN uv pip install --system --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache-dir -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "src/app.py", "--webui"]