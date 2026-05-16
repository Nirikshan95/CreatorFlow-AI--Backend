from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker, MappedAsDataclass
from datetime import datetime
import os


class Base(DeclarativeBase):
    pass

class ContentHistory(Base):
    __tablename__ = "content_history"

    id = Column(String, primary_key=True)
    video_id = Column(String, unique=True, nullable=False)
    topic = Column(String, nullable=False)
    category = Column(String, nullable=False)
    keywords = Column(JSON, nullable=False)
    title = Column(String, nullable=False)
    script_summary = Column(Text, nullable=True)
    script_data = Column(Text, nullable=True)
    seo_data = Column(JSON, nullable=True)
    community_post = Column(JSON, nullable=True)
    thumbnail_prompt = Column(JSON, nullable=True)
    marketing_strategy = Column(JSON, nullable=True)
    performance = Column(JSON, nullable=True)
    novelty_score = Column(Float, nullable=False)
    virality_score = Column(Float, nullable=False)
    critique_data = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_database_url():
    db_path = os.getenv("DATABASE_PATH", "data/content_history.db")
    return f"sqlite:///{db_path}"


engine = create_engine(get_database_url(), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
