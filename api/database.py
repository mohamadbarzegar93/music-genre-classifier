from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("POSTGRES_USER")
psswrd = os.getenv("POSTGRES_PASSWORD")
db_file = os.getenv("POSTGRES_DB")
db_host = os.getenv("DB_HOST", "localhost")

#Postgres setup
DATABASE_URL = f'postgresql://{user}:{psswrd}@{db_host}/{db_file}'
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind = engine)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String)
    predicted_genre = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=lambda:datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(engine)

def log_prediction(filename, genre, confidence):
    session = SessionLocal()
    #try/finally to ensure session is closed even if an error occurs
    try:
        prediction = Prediction(
            filename = filename,
            predicted_genre=genre,
            confidence = confidence
        )
        session.add(prediction)
        session.commit()
    finally:
        session.close()

