# Trino LCM Evaluation Environment (GPU ready for H100)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy original requirements
COPY requirements/requirements.txt /app/requirements.txt

# torch / lightning 系はベースイメージのものを使うので除外した requirements を作る
RUN grep -vE '^(torch|torchvision|torchdata|pytorch_lightning|lightning)' requirements.txt \
    > requirements-no-torch.txt

# Install Python deps (GPU 対応 PyTorch は元から入っているので触らない)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-no-torch.txt

# Install DGL for PyTorch 2.3 + CUDA 12.1
RUN pip install --no-cache-dir \
    dgl \
    -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# Copy source code
COPY . /app/

# Env
ENV PYTHONPATH=/app/src \
    DGLBACKEND=pytorch

CMD ["python", "src/trino_lcm/scripts/test_docker_status.py"]
