FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy current project files into container
COPY . .

# Run the ETL script
CMD ["python", "scrape_absences.py"]