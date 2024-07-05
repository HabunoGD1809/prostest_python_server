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

URL_BASE_DE_DATOS = "postgresql://habuno:90630898@localhost/protesta_db"
motor = create_engine(URL_BASE_DE_DATOS)
SesionLocal = sessionmaker(autocommit=False, autoflush=False, bind=motor)
Base = declarative_base()

app = FastAPI()

# Configuración de autenticación
CLAVE_SECRETA = "tu_clave_secreta"
ALGORITMO = "HS256"
MINUTOS_EXPIRACION_TOKEN = 30

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
class CrearUsuario(BaseModel):
    foto: Optional[str]
    nombre: str
    apellidos: str
    email: EmailStr
    password: str

class UsuarioSalida(BaseModel):
    id: uuid.UUID
    foto: Optional[str]
    nombre: str
    apellidos: str
    email: EmailStr

    class Config:
        from_attributes = True

class Token(BaseModel):
    token_acceso: str
    tipo_token: str

class CrearNaturaleza(BaseModel):
    nombre: str
    color: str
    icono: Optional[str]

class NaturalezaSalida(BaseModel):
    id: uuid.UUID
    nombre: str
    color: str
    icono: Optional[str]
    creado_por: uuid.UUID
    fecha_creacion: date
    soft_delete: bool

    class Config:
        from_attributes = True

class CrearCabecilla(BaseModel):
    foto: Optional[str]
    nombre: str
    apellido: str
    cedula: str
    telefono: Optional[str]
    direccion: Optional[str]

class CabecillaSalida(BaseModel):
    id: uuid.UUID
    foto: Optional[str]
    nombre: str
    apellido: str
    cedula: str
    telefono: Optional[str]
    direccion: Optional[str]
    creado_por: uuid.UUID
    fecha_creacion: date
    soft_delete: bool

    class Config:
        from_attributes = True

class CrearProtesta(BaseModel):
    nombre: str
    naturaleza_id: uuid.UUID
    provincia_id: uuid.UUID
    resumen: str
    fecha_evento: date
    cabecillas: List[uuid.UUID]

class ProtestaSalida(BaseModel):
    id: uuid.UUID
    nombre: str
    naturaleza_id: uuid.UUID
    provincia_id: uuid.UUID
    resumen: str
    fecha_evento: date
    creado_por: uuid.UUID
    fecha_creacion: date
    soft_delete: bool
    cabecillas: List[CabecillaSalida]

    class Config:
        from_attributes = True

class ProvinciaSalida(BaseModel):
    id: uuid.UUID
    nombre: str
    soft_delete: bool

    class Config:
        from_attributes = True

# Funciones auxiliares
def obtener_db():
    db = SesionLocal()
    try:
        yield db
    finally:
        db.close()

def verificar_password(password_plano, password_hash):
    return bcrypt.checkpw(password_plano.encode('utf-8'), password_hash.encode('utf-8'))

def obtener_hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def crear_token_acceso(datos: dict, delta_expiracion: Optional[timedelta] = None):
    a_codificar = datos.copy()
    if delta_expiracion:
        expira = datetime.now(timezone.utc) + delta_expiracion
    else:
        expira = datetime.now(timezone.utc) + timedelta(minutes=15)
    a_codificar.update({"exp": expira})
    token_jwt_codificado = jwt.encode(a_codificar, CLAVE_SECRETA, algorithm=ALGORITMO)
    return token_jwt_codificado

def obtener_usuario_actual(token: str = Depends(oauth2_scheme), db: Session = Depends(obtener_db)):
    excepcion_credenciales = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, CLAVE_SECRETA, algorithms=[ALGORITMO])
        email: str = payload.get("sub")
        if email is None:
            raise excepcion_credenciales
    except JWTError:
        raise excepcion_credenciales
    usuario = db.query(Usuario).filter(Usuario.email == email).first()
    if usuario is None:
        raise excepcion_credenciales
    return usuario

# Rutas de la API
@app.post("/registro", response_model=UsuarioSalida)
def registrar_usuario(usuario: CrearUsuario, db: Session = Depends(obtener_db)):
    db_usuario = db.query(Usuario).filter(Usuario.email == usuario.email).first()
    if db_usuario:
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    hash_password = obtener_hash_password(usuario.password)
    db_usuario = Usuario(
        foto=usuario.foto,
        nombre=usuario.nombre,
        apellidos=usuario.apellidos,
        email=usuario.email,
        password=hash_password
    )
    db.add(db_usuario)
    db.commit()
    db.refresh(db_usuario)
    return db_usuario

@app.post("/token", response_model=Token)
def login_para_token_acceso(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(obtener_db)):
    usuario = db.query(Usuario).filter(Usuario.email == form_data.username).first()
    if not usuario or not verificar_password(form_data.password, usuario.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expiracion_token_acceso = timedelta(minutes=MINUTOS_EXPIRACION_TOKEN)
    token_acceso = crear_token_acceso(
        datos={"sub": usuario.email}, delta_expiracion=expiracion_token_acceso
    )
    return {"token_acceso": token_acceso, "tipo_token": "bearer"}

@app.post("/naturalezas", response_model=NaturalezaSalida)
def crear_naturaleza(naturaleza: CrearNaturaleza, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_naturaleza = Naturaleza(**naturaleza.model_dump(), creado_por=usuario_actual.id)
    db.add(db_naturaleza)
    db.commit()
    db.refresh(db_naturaleza)
    return db_naturaleza

@app.post("/cabecillas", response_model=CabecillaSalida)
def crear_cabecilla(cabecilla: CrearCabecilla, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_cabecilla = Cabecilla(**cabecilla.model_dump(), creado_por=usuario_actual.id)
    db.add(db_cabecilla)
    db.commit()
    db.refresh(db_cabecilla)
    return db_cabecilla

@app.post("/protestas", response_model=ProtestaSalida)
def crear_protesta(protesta: CrearProtesta, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_protesta = Protesta(
        nombre=protesta.nombre,
        naturaleza_id=protesta.naturaleza_id,
        provincia_id=protesta.provincia_id,
        resumen=protesta.resumen,
        fecha_evento=protesta.fecha_evento,
        creado_por=usuario_actual.id
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

@app.get("/protestas", response_model=List[ProtestaSalida])
def obtener_protestas(usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    protestas = db.query(Protesta).filter(Protesta.soft_delete == False).all()
    return protestas

@app.get("/protestas/{protesta_id}", response_model=ProtestaSalida)
def obtener_protesta(protesta_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.soft_delete == False).first()
    if not protesta:
        raise HTTPException(status_code=404, detail="Protesta no encontrada")
    return protesta

@app.put("/protestas/{protesta_id}", response_model=ProtestaSalida)
def actualizar_protesta(protesta_id: uuid.UUID, protesta: CrearProtesta, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.creado_por == usuario_actual.id).first()
    if not db_protesta:
        raise HTTPException(status_code=404, detail="Protesta no encontrada o no tienes permiso para editarla")
    
    for key, value in protesta.model_dump().items():
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
def eliminar_protesta(protesta_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.creado_por == usuario_actual.id).first()
    if not db_protesta:
        raise HTTPException(status_code=404, detail="Protesta no encontrada o no tienes permiso para eliminarla")
    db_protesta.soft_delete = True
    db.commit()
    return {"detail": "Protesta eliminada exitosamente"}

@app.get("/provincias", response_model=List[ProvinciaSalida])
def obtener_provincias(db: Session = Depends(obtener_db)):
    return db.query(Provincia).filter(Provincia.soft_delete == False).all()

# Nuevas rutas para Naturalezas y Cabecillas

@app.get("/naturalezas", response_model=List[NaturalezaSalida])
def obtener_naturalezas(db: Session = Depends(obtener_db)):
    return db.query(Naturaleza).filter(Naturaleza.soft_delete == False).all()

@app.get("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def obtener_naturaleza(naturaleza_id: uuid.UUID, db: Session = Depends(obtener_db)):
    naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id, Naturaleza.soft_delete == False).first()
    if not naturaleza:
        raise HTTPException(status_code=404, detail="Naturaleza no encontrada")
    return naturaleza

@app.put("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def actualizar_naturaleza(naturaleza_id: uuid.UUID, naturaleza: CrearNaturaleza, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id, Naturaleza.creado_por == usuario_actual.id).first()
    if not db_naturaleza:
        raise HTTPException(status_code=404, detail="Naturaleza no encontrada o no tienes permiso para editarla")
    
    for key, value in naturaleza.model_dump().items():
        setattr(db_naturaleza, key, value)
    
    db.commit()
    db.refresh(db_naturaleza)
    return db_naturaleza

@app.delete("/naturalezas/{naturaleza_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_naturaleza(naturaleza_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id, Naturaleza.creado_por == usuario_actual.id).first()
    if not db_naturaleza:
        raise HTTPException(status_code=404, detail="Naturaleza no encontrada o no tienes permiso para eliminarla")
    db_naturaleza.soft_delete = True
    db.commit()
    return {"detail": "Naturaleza eliminada exitosamente"}

@app.get("/cabecillas", response_model=List[CabecillaSalida])
def obtener_cabecillas(db: Session = Depends(obtener_db)):
    return db.query(Cabecilla).filter(Cabecilla.soft_delete == False).all()

@app.get("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
def obtener_cabecilla(cabecilla_id: uuid.UUID, db: Session = Depends(obtener_db)):
    cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.soft_delete == False).first()
    if not cabecilla:
        raise HTTPException(status_code=404, detail="Cabecilla no encontrado")
    return cabecilla

@app.put("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
def actualizar_cabecilla(cabecilla_id: uuid.UUID, cabecilla: CrearCabecilla, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.creado_por == usuario_actual.id).first()
    if not db_cabecilla:
        raise HTTPException(status_code=404, detail="Cabecilla no encontrado o no tienes permiso para editarlo")
    
    for key, value in cabecilla.model_dump().items():
        setattr(db_cabecilla, key, value)
    
    db.commit()
    db.refresh(db_cabecilla)
    return db_cabecilla

@app.delete("/cabecillas/{cabecilla_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_cabecilla(cabecilla_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.creado_por == usuario_actual.id).first()
    if not db_cabecilla:
        raise HTTPException(status_code=404, detail="Cabecilla no encontrado o no tienes permiso para eliminarlo")
    db_cabecilla.soft_delete = True
    db.commit()
    return {"detail": "Cabecilla eliminado exitosamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
