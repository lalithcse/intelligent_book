#!/bin/bash

# docker build -t book-service .

docker compose up --build --force-recreate -d app
