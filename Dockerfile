# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list first to leverage Docker cache
COPY requirements.txt .

# Install dependencies first (faster rebuilds)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container (including your model)
COPY . .

# (Optional) explicitly copy model if you want to guarantee it's present
# COPY pneumonia_cnn_model.h5 /app/pneumonia_cnn_model.h5

# Expose Streamlit port
EXPOSE 8000

# Run the Streamlit app
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]

