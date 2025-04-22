FROM python:3.9.2

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# Run program
CMD ["python3", "main.py"]
