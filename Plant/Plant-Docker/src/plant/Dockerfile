FROM python:3.9.2

# Changing working directory, also creating folder
WORKDIR /workspace_pl

# Copy the rest of the code
COPY . /workspace_pl

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# Run program
CMD ["python3", "main.py"]
