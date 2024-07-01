#Backend
docker build -t shakespeare-generator-backend .
docker run -d -p 8000:8000 shakespeare-generator-backend