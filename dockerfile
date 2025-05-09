# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Increased timeout to 600 seconds (10 minutes)
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME="World"

# Run app.py when the container launches
CMD ["python", "-m", "app.main"]