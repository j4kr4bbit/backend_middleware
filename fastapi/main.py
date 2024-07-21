from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from neo4j import GraphDatabase
import requests
import os
import json

app = FastAPI()

# PostgreSQL Configuration
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, index=True)

Base.metadata.create_all(bind=engine)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Pydantic models
class ItemBase(BaseModel):
    name: str
    description: str

class ItemCreate(ItemBase):
    pass

class ItemResponse(ItemBase):
    id: int

    class Config:
        orm_mode = True

# PostgreSQL CRUD operations
@app.post("/postgres/items/", response_model=ItemResponse)
def create_postgres_item(item: ItemCreate):
    db = SessionLocal()
    db_item = Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@app.get("/postgres/items/{item_id}", response_model=ItemResponse)
def read_postgres_item(item_id: int):
    db = SessionLocal()
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item

@app.put("/postgres/items/{item_id}", response_model=ItemResponse)
def update_postgres_item(item_id: int, item: ItemCreate):
    db = SessionLocal()
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    db_item.name = item.name
    db_item.description = item.description
    db.commit()
    db.refresh(db_item)
    return db_item

@app.delete("/postgres/items/{item_id}")
def delete_postgres_item(item_id: int):
    db = SessionLocal()
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    db.delete(db_item)
    db.commit()
    return {"ok": True}

# Neo4j CRUD operations
@app.post("/neo4j/items/")
def create_neo4j_item(item: ItemCreate):
    with neo4j_driver.session() as session:
        session.run("CREATE (a:Item {name: $name, description: $description})", 
                    name=item.name, description=item.description)
    return {"ok": True}

@app.get("/neo4j/items/{name}")
def read_neo4j_item(name: str):
    with neo4j_driver.session() as session:
        result = session.run("MATCH (a:Item {name: $name}) RETURN a", name=name)
        record = result.single()
        if record is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"name": record["a"]["name"], "description": record["a"]["description"]}

@app.put("/neo4j/items/{name}")
def update_neo4j_item(name: str, item: ItemCreate):
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (a:Item {name: $name})
            SET a.name = $new_name, a.description = $description
            RETURN a
        """, name=name, new_name=item.name, description=item.description)
        record = result.single()
        if record is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"name": record["a"]["name"], "description": record["a"]["description"]}

@app.delete("/neo4j/items/{name}")
def delete_neo4j_item(name: str):
    with neo4j_driver.session() as session:
        session.run("MATCH (a:Item {name: $name}) DELETE a", name=name)
    return {"ok": True}

# Ollama Integration
class OllamaRequest(BaseModel):
    prompt: str

class OllamaResponse(BaseModel):
    response: str

@app.post("/ollama/", response_model=OllamaResponse)
def query_ollama(request: OllamaRequest):
    ollama_url = os.getenv("OLLAMA_URL")
    data = {
        "model": "llama3",
        "prompt": f"{request.prompt} + please format your response in JSON.",
        "format": "json",
        "stream": False
    }
    response = requests.post(f"{ollama_url}/api/generate", json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    try:
        response_data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="Failed to parse response from Ollama service")
    
    # Ensure response is properly formatted
    try:
        response_text = response_data.get("response", "No response key in Ollama response")
        cleaned_response = json.dumps(json.loads(response_text), indent=4)
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=500, detail="Failed to format response from Ollama service")

    return OllamaResponse(response=cleaned_response)
