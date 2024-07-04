from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, String, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import date, datetime, timedelta, timezone
import uuid
import bcrypt
import jwt

# Configuración de la base de datos
SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configuración de la aplicación
app = FastAPI()

# Configuración de autenticación
SECRET_KEY = "tu_clave_secreta"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Modelos SQLAlchemy
class Usuario(Base):
    __tablename__ = "usuarios"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    foto = Column(String)
    nombre = Column(String, nullable=False)
    apellidos = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    fecha_creacion = Column(Date, default=datetime.now(timezone.utc))
    soft_delete = Column(Boolean, default=False)

class Provincia(Base):
    __tablename__ = "provincias"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nombre = Column(String, unique=True, nullable=False)
    soft_delete = Column(Boolean, default=False)

class Naturaleza(Base):
    __tablename__ = "naturalezas"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nombre = Column(String, unique=True, nullable=False)
    color = Column(String(7), nullable=False)
    icono = Column(String)
    creado_por = Column(UUID(as_uuid=True), ForeignKey('usuarios.id'))
    fecha_creacion = Column(Date, default=datetime.now(timezone.utc))
    soft_delete = Column(Boolean, default=False)

class Cabecilla(Base):
    __tablename__ = "cabecillas"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    foto = Column(String)
    nombre = Column(String, nullable=False)
    apellido = Column(String, nullable=False)
    cedula = Column(String, unique=True, nullable=False)
    telefono = Column(String)
    direccion = Column(String)
    creado_por = Column(UUID(as_uuid=True), ForeignKey('usuarios.id'))
    fecha_creacion = Column(Date, default=datetime.now(timezone.utc))
    soft_delete = Column(Boolean, default=False)

class Protesta(Base):
    __tablename__ = "protestas"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nombre = Column(String, nullable=False)
    naturaleza_id = Column(UUID(as_uuid=True), ForeignKey('naturalezas.id'))
    provincia_id = Column(UUID(as_uuid=True), ForeignKey('provincias.id'))
    resumen = Column(String)
    fecha_evento = Column(Date, nullable=False)
    creado_por = Column(UUID(as_uuid=True), ForeignKey('usuarios.id'))
    fecha_creacion = Column(Date, default=datetime.now(timezone.utc))
    soft_delete = Column(Boolean, default=False)
    cabecillas = relationship("Cabecilla", secondary="protestas_cabecillas")

class ProtestaCabecilla(Base):
    __tablename__ = "protestas_cabecillas"
    protesta_id = Column(UUID(as_uuid=True), ForeignKey('protestas.id'), primary_key=True)
    cabecilla_id = Column(UUID(as_uuid=True), ForeignKey('cabecillas.id'), primary_key=True)

# Modelos Pydantic para la API
class UsuarioCreate(BaseModel):
    foto: Optional[str]
    nombre: str
    apellidos: str
    email: EmailStr
    password: str

class UsuarioOut(BaseModel):
    id: uuid.UUID
    foto: Optional[str]
    nombre: str
    apellidos: str
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class NaturalezaCreate(BaseModel):
    nombre: str
    color: str
    icono: Optional[str]

class CabecillaCreate(BaseModel):
    foto: Optional[str]
    nombre: str
    apellido: str
    cedula: str
    telefono: Optional[str]
    direccion: Optional[str]

class ProtestaCreate(BaseModel):
    nombre: str
    naturaleza_id: uuid.UUID
    provincia_id: uuid.UUID
    resumen: str
    fecha_evento: date
    cabecillas: List[uuid.UUID]

# Funciones auxiliares
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(Usuario).filter(Usuario.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# Rutas de la API
# Continuación de main.py

@app.post("/register", response_model=UsuarioOut)
def register_user(user: UsuarioCreate, db: Session = Depends(get_db)):
    db_user = db.query(Usuario).filter(Usuario.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = Usuario(
        foto=user.foto,
        nombre=user.nombre,
        apellidos=user.apellidos,
        email=user.email,
        password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/naturalezas", response_model=NaturalezaCreate)
def create_naturaleza(naturaleza: NaturalezaCreate, current_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    db_naturaleza = Naturaleza(**naturaleza.model_dump(), creado_por=current_user.id)
    db.add(db_naturaleza)
    db.commit()
    db.refresh(db_naturaleza)
    return db_naturaleza

@app.post("/cabecillas", response_model=CabecillaCreate)
def create_cabecilla(cabecilla: CabecillaCreate, current_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    db_cabecilla = Cabecilla(**cabecilla.model_dump(), creado_por=current_user.id)
    db.add(db_cabecilla)
    db.commit()
    db.refresh(db_cabecilla)
    return db_cabecilla

@app.post("/protestas", response_model=ProtestaCreate)
def create_protesta(protesta: ProtestaCreate, current_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    db_protesta = Protesta(
        nombre=protesta.nombre,
        naturaleza_id=protesta.naturaleza_id,
        provincia_id=protesta.provincia_id,
        resumen=protesta.resumen,
        fecha_evento=protesta.fecha_evento,
        creado_por=current_user.id
    )
    db.add(db_protesta)
    db.commit()
    db.refresh(db_protesta)
    
    for cabecilla_id in protesta.cabecillas:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id).first()
        if db_cabecilla:
            db_protesta.cabecillas.append(db_cabecilla)
    
    db.commit()
    return db_protesta

@app.get("/protestas", response_model=List[ProtestaCreate])
def get_protestas(current_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    protestas = db.query(Protesta).filter(Protesta.soft_delete == False).all()
    return protestas

@app.get("/protestas/{protesta_id}", response_model=ProtestaCreate)
def get_protesta(protesta_id: uuid.UUID, current_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.soft_delete == False).first()
    if not protesta:
        raise HTTPException(status_code=404, detail="Protesta not found")
    return protesta

@app.put("/protestas/{protesta_id}", response_model=ProtestaCreate)
def update_protesta(protesta_id: uuid.UUID, protesta: ProtestaCreate, current_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.creado_por == current_user.id).first()
    if not db_protesta:
        raise HTTPException(status_code=404, detail="Protesta not found or you don't have permission to edit")
    
    for key, value in protesta.dict().items():
        setattr(db_protesta, key, value)
    
    db_protesta.cabecillas.clear()
    for cabecilla_id in protesta.cabecillas:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id).first()
        if db_cabecilla:
            db_protesta.cabecillas.append(db_cabecilla)
    
    db.commit()
    db.refresh(db_protesta)
    return db_protesta

@app.delete("/protestas/{protesta_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_protesta(protesta_id: uuid.UUID, current_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.creado_por == current_user.id).first()
    if not db_protesta:
        raise HTTPException(status_code=404, detail="Protesta not found or you don't have permission to delete")
    db_protesta.soft_delete = True
    db.commit()
    return {"detail": "Protesta deleted successfully"}

@app.get("/provincias", response_model=List[Provincia])
def get_provincias(db: Session = Depends(get_db)):
    return db.query(Provincia).filter(Provincia.soft_delete == False).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
