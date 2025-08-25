FROM python:3.9-slim

WORKDIR /app

RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get update --fix-missing \
 && apt-get install -y --no-install-recommends gcc build-essential \ 
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt . 
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove gcc build-essential

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app2.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
