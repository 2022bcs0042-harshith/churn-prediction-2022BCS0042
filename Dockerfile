# ENV PYTHONPATH=/app
# # Use official Python image 
# FROM python:3.11-slim 
# # Set working directory 
# WORKDIR /app 
# # Copy requirements first (for better caching) 
# COPY requirements.txt . 
# # Install dependencies 
# RUN pip install --no-cache-dir -r requirements.txt 
# # Copy application code 
# COPY app/ ./app/ 
# # Expose port 
# EXPOSE 8000 
# # Run the application 
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 

FROM python:3.11-slim

WORKDIR /app

# Set Python path properly
ENV PYTHONPATH=/app

# Copy entire project (IMPORTANT FIX)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]