# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies and git-lfs (in case LFS is used)
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Copy dependency list first to leverage Docker cache
COPY requirements.txt .

# Install dependencies first (faster rebuilds)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Ensure LFS files (like .h5 model) are actually pulled if used
RUN git lfs pull || echo "No LFS files to pull"

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
