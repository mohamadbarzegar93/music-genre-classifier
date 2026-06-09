from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

#SQLAlchemy setup
DATABASE_URL = "sqlite:///./predictions.db"
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

