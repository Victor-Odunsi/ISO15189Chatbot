FROM python:3.11.9-slim

# Lambda Web Adapter: proxies Lambda invokes (including RESPONSE_STREAM
# mode, configured on the function's Function URL + AWS_LWA_INVOKE_MODE)
# to the normal uvicorn process below -- no ASGI/Lambda handler code
# needed, api/chat.py's StreamingResponse works unmodified.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app/backend

RUN pip install --upgrade pip

COPY backend/requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

# Migrations are a deploy-time step (run `alembic upgrade head` against
# the RDS endpoint after `sam deploy`), not run here -- doing it on every
# cold start adds latency and risks concurrent-cold-start migration races.
# Same entrypoint locally and on Lambda -- LWA (added above) is configured
# via AWS_LWA_PORT=8000 (see template.yaml) to proxy to this port rather
# than its own default of 8080, so nothing here needs to change per target.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
