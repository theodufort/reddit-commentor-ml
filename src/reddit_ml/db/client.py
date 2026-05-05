import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.environ["PG_DATABASE_NAME"],
    host=os.environ["PG_DATABASE_HOST"],
    port=os.environ["PG_DATABASE_PORT"],
    user=os.environ["PG_DATABASE_USER"],
    password=os.environ["PG_DATABASE_PASS"],
)
