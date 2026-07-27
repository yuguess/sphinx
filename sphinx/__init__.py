import os
from dotenv import load_dotenv

ENV_FILE = os.environ.get('ENV_FILE', None)
print(f"load ENV_FILE:{ENV_FILE}")
load_dotenv(ENV_FILE, override=True)