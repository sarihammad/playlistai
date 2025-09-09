FROM rayproject/ray-ml:2.35.0-py311

# Install additional dependencies
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers

# Set working directory
WORKDIR /workspace

# Copy core library and install in development mode
COPY core/ ./core/
RUN pip install -e ./core

# Copy training scripts
COPY core/train/ ./core/train/

# Default command
CMD ["python", "-m", "core.train.ray_train", "--config", "core/train/config.yaml"]

