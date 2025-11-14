FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Саначала копируем только requirments.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Затем копируем остальной код
COPY . .

EXPOSE 5000

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "depositsweb:app"]
