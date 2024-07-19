# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Create and set the working directory
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Make the start.sh script executable
RUN chmod +x start.sh

# Expose the port the app runs on
EXPOSE 5000

# Define environment variables
ENV OPENAI_API_KEY=Your_API_Key
ENV LANGCHAIN_API_KEY=Your_API_Key
ENV ELEVEN_API_KEY=Your_API_Key
ENV HF_TOKEN=Your_API_Key

# Set the entry point to run the Flask application
ENTRYPOINT ["sh", "start.sh"]
