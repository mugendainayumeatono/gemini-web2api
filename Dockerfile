FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY gemini_web2api/ ./gemini_web2api/
RUN cp -n gemini_web2api/config.example.json gemini_web2api/config.json 2>/dev/null || true
EXPOSE 8081

CMD ["python", "gemini_web2api/main.py"]
