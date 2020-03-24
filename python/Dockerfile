ARG PYTHON_TAG

FROM python:${PYTHON_TAG}

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt && rm requirements.txt
