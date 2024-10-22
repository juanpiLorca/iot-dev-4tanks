# Redis Pub/Sub Data Bus 

This project demonstrates a Redis-based publish-subscribe data bus system in real-time. The system involves three main components:
1. **Client**: Publishes raw process data to a Redis channel (`plant_outputs`).
2. **Autoencoder**: Subscribes to the data from the `plant_outputs` channel, processes it, and publishes the transformed data to another channel (`plant_outputs_filtered`).
3. **Controller**: Subscribes to the data from the `plant_outputs_filered` channel, processes it, and publishes the transformed data to another channel (`plant_inputs`)

## Prerequisites

Ensure you have the following installed:
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- Python 3.x
- Redis Python Client: `redis-py`

You can install the Redis client using pip:
```bash
pip install redis numpy
```

To run the project (Windows): 

1. **Run docker desktop**
2. **Run the following commands:**

```bash
docker-compose up

python3 Client_databus.py
python3 AE_databus.py
python3 Controller_databus.py
```

