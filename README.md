# docker-for-classifier-app

For you machine have the following steps:
Create a docker-compose.yml file with the following content:
'''
version: '3.8'

services:
  frontend:
    image: your-dockerhub-username/potato-disease-frontend:latest
    ports:
      - "3000:80"
    depends_on:
      - backend

  backend:
    image: your-dockerhub-username/potato-disease-backend:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models

networks:
  default:
    name: potato-disease-network

'''

Then :
'''
docker-compose pull
'''

&

'''
docker-compose up 
'''
