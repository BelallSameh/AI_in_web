# Use a base image with Python
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Define environment variable for database URL
ENV DATABASE_URL=postgresql://user:password@db:5432/mydatabase

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]