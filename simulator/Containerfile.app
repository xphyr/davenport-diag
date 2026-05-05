FROM quay.io/fedora/fedora-minimal:latest
RUN mkdir /app && mkdir /models
COPY static /app/static
COPY templates /app/templates
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt
RUN microdnf install python3 python3-pip libgomp -y && microdnf clean all
RUN pip install -r /app/requirements.txt
EXPOSE 5000
WORKDIR /app
CMD ["python", "app.py"]