# Pong Hawk

## Production

To run these app including Nginx, Gunicorn and everything:

```sh
docker compose up
```

Check that server is online at http://localhost:9000/api/hello

## Development

Run the services you need, and then develop with poetry

```sh
# Create folders
mkdir db static storage

# Services
docker compose up postgres

# App
cp .env.sample .env
poetry install
poetry run python manage.py runserver
```
