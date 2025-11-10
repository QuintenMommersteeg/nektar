
#!/bin/sh
set -e
# Seed the demo DB only if it doesn't exist
if [ ! -f /app/demo.db ]; then
  echo "Seeding demo database..."
  python /app/seed_db.py
fi

# Start the API
exec uvicorn app:app --host 0.0.0.0 --port 8000
