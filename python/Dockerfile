ARG PYTHON_TAG

FROM python:${PYTHON_TAG}

ENV PYTHON_PACKAGES="\
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    nltk \
    seaborn \
    keras \
    redis \
    hiredis \
    influxdb \
" 

RUN pip install --upgrade pip && pip install --no-cache-dir $PYTHON_PACKAGES
