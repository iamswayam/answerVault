from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
from app.config import DATABASE_URL

from pgvector.sqlalchemy import Vector


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)      # "user" or "model"
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class QAEntry(Base):
    __tablename__ = "qa_entries"

    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)