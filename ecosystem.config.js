module.exports = {
  apps: [
    {
      name: "ocr_backend",
      // On Windows, run uvicorn via Python module to avoid locating a script
      script: "python",
      args: "-m uvicorn app.main:app --host 0.0.0.0 --port 8000",
      // Use the project's virtualenv Python explicitly on Windows
      interpreter: "C:/Users/hp/Desktop/Work/python/new-repo/.venv/Scripts/python.exe",
      cwd: "C:/Users/hp/Desktop/Work/python/new-repo",
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "512M",
      env: {
        ENVIRONMENT: "production",
        PORT: 8000,
        // Optional: set if you later re-enable env URL in config
        // DATABASE_URL: "postgresql+psycopg://postgres:postgres@localhost:5433/ocr_db"
      }
    }
  ]
};
