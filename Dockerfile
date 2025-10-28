# Dockerfile
# TODO: Copy content from artifact: docker_setup
# This is a placeholder - replace with actual Dockerfile content

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY autonomous_trading_bot/ /app/autonomous_trading_bot/
COPY *.py /app/
COPY config.json /app/

CMD ["python", "paper_live_trading.py"]
