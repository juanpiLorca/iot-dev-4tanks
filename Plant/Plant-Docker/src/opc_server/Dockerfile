FROM python:3.9.2

# Changing working directory, also creating folder
WORKDIR /workspace_sv

# Copy the rest of the code
COPY . /workspace_sv

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# Exposing port where OPC Plant server will be running
EXPOSE 4848

# Run program
CMD ["python3", "main.py"]
