import asyncio
import os
import shutil
from signal import signal
from fastapi import FastAPI, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, String, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import List, Optional
from datetime import date, datetime, timedelta, timezone
from colorama import init, Fore, Style
from sqlalchemy.exc import IntegrityError
import psycopg2
import uuid
import bcrypt
import sys
import uvicorn
import os
import shutil
import imghdr
from fastapi import UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import joinedload


init(autoreset=True)

usuario = "habuno"
contraseña = "90630898"
base_de_datos = "protestas_db"

try:
    conn = psycopg2.connect(f"postgresql://{usuario}:{contraseña}@localhost/{base_de_datos}")
    print(Fore.BLUE + "Conexión exitosa a la base de datos" + Style.RESET_ALL)
    conn.close()
except Exception as e:
    print(Fore.RED + f"Error al conectar a la base de datos: {str(e)}" + Style.RESET_ALL)
    sys.exit(1)

URL_BASE_DE_DATOS = f"postgresql://{usuario}:{contraseña}@localhost/{base_de_datos}"
motor = create_engine(URL_BASE_DE_DATOS)
SesionLocal = sessionmaker(autocommit=False, autoflush=False, bind=motor)
Base = declarative_base()

app = FastAPI()

# Configuracion CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# Configuración de autenticación
CLAVE_SECRETA = "b1T!2F3h6kJ8mN9pQ1rT3vW7yZ$0aE#4"
ALGORITMO = "HS256"
MINUTOS_EXPIRACION_TOKEN_ACCESO = 30
DIAS_EXPIRACION_TOKEN_ACTUALIZACION = 7

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
    fecha_creacion = Column(Date, default=date.today())
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
    fecha_creacion = Column(Date, default=date.today())
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
    fecha_creacion = Column(Date, default=date.today())
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
    fecha_creacion = Column(Date, default=date.today())
    soft_delete = Column(Boolean, default=False)
    
    naturaleza = relationship("Naturaleza")
    provincia = relationship("Provincia")
    cabecillas = relationship("Cabecilla", secondary="protestas_cabecillas")

class ProtestaCabecilla(Base):
    __tablename__ = "protestas_cabecillas"
    protesta_id = Column(UUID(as_uuid=True), ForeignKey('protestas.id'), primary_key=True)
    cabecilla_id = Column(UUID(as_uuid=True), ForeignKey('cabecillas.id'), primary_key=True)

# Modelos Pydantic para la API
class CrearUsuario(BaseModel):
    foto: Optional[str] = None
    nombre: str
    apellidos: str
    email: EmailStr
    password: str
    repetir_password: str

    @field_validator('repetir_password')
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Las contraseñas no coinciden')
        return v

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
    token_actualizacion: str
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

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("fecha_creacion"), datetime):
                obj["fecha_creacion"] = obj["fecha_creacion"].date()
        return super().model_validate(obj)

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
    fecha_creacion: date = Field(..., alias="fecha_creacion")
    soft_delete: bool

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.date().isoformat()
        }

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj.get("fecha_creacion"), datetime):
            obj["fecha_creacion"] = obj["fecha_creacion"].date()
        return super().model_validate(obj)

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
    fecha_creacion: date = Field(..., alias="fecha_creacion")
    soft_delete: bool
    cabecillas: List[CabecillaSalida]

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.date().isoformat()
        }

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("fecha_creacion"), datetime):
                obj["fecha_creacion"] = obj["fecha_creacion"].date()
        return super().model_validate(obj)

class ProvinciaSalida(BaseModel):
    id: uuid.UUID
    nombre: str
    soft_delete: bool

    class Config:
        from_attributes = True

class NuevoCabecilla(BaseModel):
    nombre: str
    apellido: str
    cedula: str
    telefono: Optional[str]
    direccion: Optional[str]

class NuevaNaturaleza(BaseModel):
    nombre: str
    color: str
    icono: Optional[str]

class CrearProtestaCompleta(BaseModel):
    nombre: str
    naturaleza_id: Optional[uuid.UUID]
    nueva_naturaleza: Optional[NuevaNaturaleza]
    provincia_id: uuid.UUID
    resumen: str
    fecha_evento: date
    cabecillas: List[uuid.UUID]
    nuevos_cabecillas: List[NuevoCabecilla]

class ResumenPrincipal(BaseModel):
    total_protestas: int
    protestas_recientes: List[ProtestaSalida]

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
    try:
        payload = jwt.decode(token, CLAVE_SECRETA, algorithms=[ALGORITMO])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado"
        )
    
    usuario = db.query(Usuario).filter(Usuario.email == email).first()
    if usuario is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado"
        )
    return usuario

def paginar(query, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size
    }

def crear_token(datos: dict, tiempo_expiracion: timedelta):
    a_codificar = datos.copy()
    expira = datetime.utcnow() + tiempo_expiracion
    a_codificar.update({"exp": expira})
    token_codificado = jwt.encode(a_codificar, CLAVE_SECRETA, algorithm=ALGORITMO)
    return token_codificado

def crear_tokens(datos: dict):
    token_acceso = crear_token(datos, timedelta(minutes=MINUTOS_EXPIRACION_TOKEN_ACCESO))
    token_actualizacion = crear_token(datos, timedelta(days=DIAS_EXPIRACION_TOKEN_ACTUALIZACION))
    return token_acceso, token_actualizacion

# Rutas de la API
@app.get("/usuarios/me", response_model=UsuarioSalida)
def obtener_usuario_actual(usuario_actual: Usuario = Depends(obtener_usuario_actual)):
    return usuario_actual

@app.post("/registro", response_model=UsuarioSalida)
async def registrar_usuario(
    nombre: str = Form(...),
    apellidos: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    repetir_password: str = Form(...),
    foto: UploadFile = File(None),
    db: Session = Depends(obtener_db)
):
    if password != repetir_password:
        raise HTTPException(status_code=400, detail="Las contraseñas no coinciden")

    try:
        hash_password = obtener_hash_password(password)
        
        # Guardar la foto si se proporciona
        foto_path = None
        if foto:
            validate_image(foto)
            file_name = f"usuario_{uuid.uuid4()}_{foto.filename}"
            file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
            save_upload_file(foto, file_path)
            foto_path = file_path

        db_usuario = Usuario(
            foto=foto_path,
            nombre=nombre,
            apellidos=apellidos,
            email=email,
            password=hash_password
        )
        db.add(db_usuario)
        db.commit()
        db.refresh(db_usuario)
        print(Fore.GREEN + f"Usuario registrado exitosamente: {email}" + Style.RESET_ALL)
        return db_usuario
    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "usuarios_email_key" in str(e.orig):
                print(Fore.YELLOW + f"Intento de registrar usuario con email duplicado: {email}" + Style.RESET_ALL)
                raise HTTPException(
                    status_code=400,
                    detail=f"Ya existe un usuario con el email '{email}'"
                )
        print(Fore.RED + f"Error de integridad al registrar usuario: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(
            status_code=400,
            detail="Error al registrar el usuario. Por favor, intente de nuevo."
        )
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al registrar usuario: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/token", response_model=Token)
def login_para_token_acceso(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(obtener_db)):
    usuario = db.query(Usuario).filter(Usuario.email == form_data.username).first()
    if not usuario or not verificar_password(form_data.password, usuario.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_acceso, token_actualizacion = crear_tokens({"sub": usuario.email})
    return {"token_acceso": token_acceso, "token_actualizacion": token_actualizacion, "tipo_token": "bearer"}

@app.post("/token/renovar", response_model=Token)
def renovar_token(token: str = Depends(oauth2_scheme), db: Session = Depends(obtener_db)):
    try:
        payload = jwt.decode(token, CLAVE_SECRETA, algorithms=[ALGORITMO])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=400, detail="Token inválido")
        usuario = db.query(Usuario).filter(Usuario.email == email).first()
        if usuario is None:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        nuevo_token_acceso, nuevo_token_actualizacion = crear_tokens({"sub": usuario.email})
        return {"token_acceso": nuevo_token_acceso, "token_actualizacion": nuevo_token_actualizacion, "tipo_token": "bearer"}
    except JWTError:
        raise HTTPException(status_code=400, detail="Token inválido o expirado")

@app.get("/pagina-principal", response_model=ResumenPrincipal)
def obtener_resumen_principal(usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    total_protestas = db.query(Protesta).filter(Protesta.soft_delete == False).count()
    protestas_recientes = db.query(Protesta).filter(Protesta.soft_delete == False).order_by(Protesta.fecha_creacion.desc()).limit(5).all()
    
    return ResumenPrincipal(
        total_protestas=total_protestas,
        protestas_recientes=protestas_recientes
    )

@app.post("/naturalezas", response_model=NaturalezaSalida)
def crear_naturaleza(naturaleza: CrearNaturaleza, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_naturaleza = Naturaleza(**naturaleza.model_dump(), creado_por=usuario_actual.id)
        db.add(db_naturaleza)
        db.commit()
        db.refresh(db_naturaleza)
        print(Fore.GREEN + f"Naturaleza creada exitosamente: {naturaleza.nombre}" + Style.RESET_ALL)
        return db_naturaleza
    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "naturalezas_nombre_key" in str(e.orig):
                print(Fore.YELLOW + f"Intento de crear naturaleza con nombre duplicado: {naturaleza.nombre}" + Style.RESET_ALL)
                raise HTTPException(
                    status_code=400,
                    detail=f"Ya existe una naturaleza con el nombre '{naturaleza.nombre}'"
                )
        print(Fore.RED + f"Error de integridad al crear naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(
            status_code=400,
            detail="Error al crear la naturaleza. Por favor, intente de nuevo."
        )
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al crear naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/cabecillas", response_model=CabecillaSalida)
def crear_cabecilla(cabecilla: CrearCabecilla, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_cabecilla = Cabecilla(**cabecilla.model_dump(), creado_por=usuario_actual.id)
        db.add(db_cabecilla)
        db.commit()
        db.refresh(db_cabecilla)
        print(Fore.GREEN + f"Cabecilla creado exitosamente: {cabecilla.nombre} {cabecilla.apellido}" + Style.RESET_ALL)
        return db_cabecilla
    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "cabecillas_cedula_key" in str(e.orig):
                print(Fore.YELLOW + f"Intento de crear cabecilla con cédula duplicada: {cabecilla.cedula}" + Style.RESET_ALL)
                raise HTTPException(
                    status_code=400,
                    detail=f"Ya existe un cabecilla con la cédula '{cabecilla.cedula}'"
                )
        print(Fore.RED + f"Error de integridad al crear cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(
            status_code=400,
            detail="Error al crear el cabecilla. Por favor, intente de nuevo."
        )
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al crear cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/protestas/completa", response_model=ProtestaSalida)
def crear_protesta_completa(protesta: CrearProtestaCompleta, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    # Crear nueva naturaleza si se proporciona
    if protesta.nueva_naturaleza:
        nueva_naturaleza = Naturaleza(**protesta.nueva_naturaleza.model_dump(), creado_por=usuario_actual.id)
        db.add(nueva_naturaleza)
        db.flush()
        naturaleza_id = nueva_naturaleza.id
    else:
        naturaleza_id = protesta.naturaleza_id

    # Crear nuevos cabecillas
    nuevos_cabecillas_ids = []
    for nuevo_cabecilla in protesta.nuevos_cabecillas:
        db_cabecilla = Cabecilla(**nuevo_cabecilla.model_dump(), creado_por=usuario_actual.id)
        db.add(db_cabecilla)
        db.flush()
        nuevos_cabecillas_ids.append(db_cabecilla.id)

    # Crear la protesta
    db_protesta = Protesta(
        nombre=protesta.nombre,
        naturaleza_id=naturaleza_id,
        provincia_id=protesta.provincia_id,
        resumen=protesta.resumen,
        fecha_evento=protesta.fecha_evento,
        creado_por=usuario_actual.id
    )
    db.add(db_protesta)
    db.flush()

    # Asociar cabecillas existentes y nuevos
    for cabecilla_id in protesta.cabecillas + nuevos_cabecillas_ids:
        db_protesta.cabecillas.append(db.query(Cabecilla).get(cabecilla_id))

    db.commit()
    db.refresh(db_protesta)
    return db_protesta

@app.post("/protestas", response_model=ProtestaSalida)
def crear_protesta(protesta: CrearProtesta, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
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
        print(Fore.GREEN + f"Protesta creada exitosamente: {protesta.nombre}" + Style.RESET_ALL)
        return db_protesta
    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            print(Fore.YELLOW + f"Error de unicidad al crear protesta: {str(e)}" + Style.RESET_ALL)
            raise HTTPException(
                status_code=400,
                detail="Error de unicidad al crear la protesta. Por favor, verifique los datos e intente de nuevo."
            )
        print(Fore.RED + f"Error de integridad al crear protesta: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(
            status_code=400,
            detail="Error al crear la protesta. Por favor, intente de nuevo."
        )
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al crear protesta: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    

@app.get("/protestas", response_model=List[ProtestaSalida])
def obtener_protestas(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    fecha_desde: Optional[date] = None,
    fecha_hasta: Optional[date] = None,
    provincia_id: Optional[uuid.UUID] = None,
    naturaleza_id: Optional[uuid.UUID] = None,
    db: Session = Depends(obtener_db),
    usuario_actual: Usuario = Depends(obtener_usuario_actual)
):
    query = db.query(Protesta).filter(Protesta.soft_delete == False)
    
    if fecha_desde:
        query = query.filter(Protesta.fecha_evento >= fecha_desde)
    if fecha_hasta:
        query = query.filter(Protesta.fecha_evento <= fecha_hasta)
    if provincia_id:
        query = query.filter(Protesta.provincia_id == provincia_id)
    if naturaleza_id:
        query = query.filter(Protesta.naturaleza_id == naturaleza_id)
    
    total = query.count()
    protestas = query.options(joinedload(Protesta.cabecillas)).offset((page - 1) * page_size).limit(page_size).all()
    
    return protestas

@app.get("/protestas/{protesta_id}", response_model=ProtestaSalida)
def obtener_protesta(protesta_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.soft_delete == False).first()
        if not protesta:
            print(Fore.YELLOW + f"Protesta no encontrada: {protesta_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Protesta no encontrada")
        print(Fore.GREEN + f"Protesta obtenida exitosamente: {protesta.nombre}" + Style.RESET_ALL)
        return protesta
    except HTTPException as he:
        raise he
    except Exception as e:
        print(Fore.RED + f"Error al obtener protesta: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/protestas/{protesta_id}", response_model=ProtestaSalida)
def actualizar_protesta(protesta_id: uuid.UUID, protesta: CrearProtesta, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.creado_por == usuario_actual.id).first()
        if not db_protesta:
            print(Fore.YELLOW + f"Protesta no encontrada o sin permisos: {protesta_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Protesta no encontrada o no tienes permiso para editarla")
        
        # Actualizar campos simples
        db_protesta.nombre = protesta.nombre
        db_protesta.resumen = protesta.resumen
        db_protesta.fecha_evento = protesta.fecha_evento
        
        # Actualizar relaciones con objetos completos en lugar de IDs
        db_protesta.naturaleza = db.query(Naturaleza).get(protesta.naturaleza_id)
        db_protesta.provincia = db.query(Provincia).get(protesta.provincia_id)
        
        # Actualizar cabecillas
        db_protesta.cabecillas = []
        for cabecilla_id in protesta.cabecillas:
            db_cabecilla = db.query(Cabecilla).get(cabecilla_id)
            if db_cabecilla:
                db_protesta.cabecillas.append(db_cabecilla)
        
        db.commit()
        db.refresh(db_protesta)
        print(Fore.GREEN + f"Protesta actualizada exitosamente: {db_protesta.nombre}" + Style.RESET_ALL)
        return db_protesta
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al actualizar protesta: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


@app.delete("/protestas/{protesta_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_protesta(protesta_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id, Protesta.creado_por == usuario_actual.id).first()
        if not db_protesta:
            print(Fore.YELLOW + f"Protesta no encontrada o sin permisos para eliminar: {protesta_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Protesta no encontrada o no tienes permiso para eliminarla")
        db_protesta.soft_delete = True
        db.commit()
        print(Fore.GREEN + f"Protesta eliminada exitosamente: {db_protesta.nombre}" + Style.RESET_ALL)
        return {"detail": "Protesta eliminada exitosamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al eliminar protesta: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/provincias", response_model=List[ProvinciaSalida])
def obtener_provincias(db: Session = Depends(obtener_db)):
    try:
        provincias = db.query(Provincia).filter(Provincia.soft_delete == False).all()
        print(Fore.GREEN + f"Provincias obtenidas exitosamente. Total: {len(provincias)}" + Style.RESET_ALL)
        return provincias
    except Exception as e:
        print(Fore.RED + f"Error al obtener provincias: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/naturalezas", response_model=List[NaturalezaSalida])
def obtener_naturalezas(db: Session = Depends(obtener_db)):
    try:
        naturalezas = db.query(Naturaleza).filter(Naturaleza.soft_delete == False).all()
        print(Fore.GREEN + f"Naturalezas obtenidas exitosamente. Total: {len(naturalezas)}" + Style.RESET_ALL)
        return [NaturalezaSalida.model_validate(n.__dict__) for n in naturalezas]
    except Exception as e:
        print(Fore.RED + f"Error al obtener naturalezas: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def obtener_naturaleza(naturaleza_id: uuid.UUID, db: Session = Depends(obtener_db)):
    try:
        naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id, Naturaleza.soft_delete == False).first()
        if not naturaleza:
            print(Fore.YELLOW + f"Naturaleza no encontrada: {naturaleza_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")
        print(Fore.GREEN + f"Naturaleza obtenida exitosamente: {naturaleza.nombre}" + Style.RESET_ALL)
        return naturaleza
    except HTTPException as he:
        raise he
    except Exception as e:
        print(Fore.RED + f"Error al obtener naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def actualizar_naturaleza(naturaleza_id: uuid.UUID, naturaleza: CrearNaturaleza, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id, Naturaleza.creado_por == usuario_actual.id).first()
        if not db_naturaleza:
            print(Fore.YELLOW + f"Naturaleza no encontrada o sin permisos: {naturaleza_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada o no tienes permiso para editarla")
        
        for key, value in naturaleza.model_dump().items():
            setattr(db_naturaleza, key, value)
        
        db.commit()
        db.refresh(db_naturaleza)
        print(Fore.GREEN + f"Naturaleza actualizada exitosamente: {db_naturaleza.nombre}" + Style.RESET_ALL)
        return db_naturaleza
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al actualizar naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/naturalezas/{naturaleza_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_naturaleza(naturaleza_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id, Naturaleza.creado_por == usuario_actual.id).first()
        if not db_naturaleza:
            print(Fore.YELLOW + f"Naturaleza no encontrada o sin permisos para eliminar: {naturaleza_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada o no tienes permiso para eliminarla")
        db_naturaleza.soft_delete = True
        db.commit()
        print(Fore.GREEN + f"Naturaleza eliminada exitosamente: {db_naturaleza.nombre}" + Style.RESET_ALL)
        return {"detail": "Naturaleza eliminada exitosamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al eliminar naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/cabecillas", response_model=List[CabecillaSalida])
def obtener_cabecillas(db: Session = Depends(obtener_db)):
    try:
        cabecillas = db.query(Cabecilla).filter(Cabecilla.soft_delete == False).all()
        print(Fore.GREEN + f"Cabecillas obtenidos exitosamente. Total: {len(cabecillas)}" + Style.RESET_ALL)
        return cabecillas
    except Exception as e:
        print(Fore.RED + f"Error al obtener cabecillas: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
def obtener_cabecilla(cabecilla_id: uuid.UUID, db: Session = Depends(obtener_db)):
    try:
        cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.soft_delete == False).first()
        if not cabecilla:
            print(Fore.YELLOW + f"Cabecilla no encontrado: {cabecilla_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")
        print(Fore.GREEN + f"Cabecilla obtenido exitosamente: {cabecilla.nombre} {cabecilla.apellido}" + Style.RESET_ALL)
        return cabecilla
    except HTTPException as he:
        raise he
    except Exception as e:
        print(Fore.RED + f"Error al obtener cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
def actualizar_cabecilla(cabecilla_id: uuid.UUID, cabecilla: CrearCabecilla, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.creado_por == usuario_actual.id).first()
        if not db_cabecilla:
            print(Fore.YELLOW + f"Cabecilla no encontrado o sin permisos: {cabecilla_id}" + Style.RESET_ALL)
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado o no tienes permiso para editarlo")
        
        for key, value in cabecilla.model_dump().items():
            setattr(db_cabecilla, key, value)
        
        db.commit()
        db.refresh(db_cabecilla)
        print(Fore.GREEN + f"Cabecilla actualizado exitosamente: {db_cabecilla.nombre} {db_cabecilla.apellido}" + Style.RESET_ALL)
        return db_cabecilla
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al actualizar cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/cabecillas/{cabecilla_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_cabecilla(cabecilla_id: uuid.UUID, usuario_actual: Usuario = Depends(obtener_usuario_actual), db: Session = Depends(obtener_db)):
    try:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.creado_por == usuario_actual.id).first()
        if not db_cabecilla:
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado o no tienes permiso para eliminarlo")
        
        # Verificar si el cabecilla está asignado a alguna protesta
        protestas_asociadas = db.query(ProtestaCabecilla).filter(ProtestaCabecilla.cabecilla_id == cabecilla_id).first()
        if protestas_asociadas:
            raise HTTPException(status_code=400, detail="No se puede eliminar un cabecilla asignado a protestas")
        
        db_cabecilla.soft_delete = True
        db.commit()
        print(Fore.GREEN + f"Cabecilla eliminado exitosamente: {db_cabecilla.nombre} {db_cabecilla.apellido}" + Style.RESET_ALL)
        return {"detail": "Cabecilla eliminado exitosamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al eliminar cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

UPLOAD_DIRECTORY = "uploads"
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_IMAGE_TYPES = ['jpeg', 'png', 'gif']

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def validate_image(file: UploadFile):
    # Verificar el tamaño del archivo
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail=f"El archivo es demasiado grande. El tamaño máximo es de {MAX_IMAGE_SIZE // (1024 * 1024)} MB.")
    
    # Verificar el tipo de archivo
    contents = file.file.read()
    file.file.seek(0)
    file_type = imghdr.what(None, h=contents)
    if file_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido. Solo se aceptan {', '.join(ALLOWED_IMAGE_TYPES)}.")

def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return destination

@app.post("/usuarios/foto", response_model=UsuarioSalida)
async def actualizar_foto_usuario(
    foto: UploadFile = File(...),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db)
):
    validate_image(foto)
    file_name = f"{usuario_actual.id}_{foto.filename}"
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    save_upload_file(foto, file_path)
    
    usuario_actual.foto = file_path
    db.commit()
    return usuario_actual

@app.post("/cabecillas/{cabecilla_id}/foto", response_model=CabecillaSalida)
async def actualizar_foto_cabecilla(
    cabecilla_id: uuid.UUID,
    foto: UploadFile = File(...),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db)
):
    cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.creado_por == usuario_actual.id).first()
    if not cabecilla:
        raise HTTPException(status_code=404, detail="Cabecilla no encontrado o no tienes permiso para editarlo")
    
    validate_image(foto)
    file_name = f"cabecilla_{cabecilla_id}_{foto.filename}"
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    save_upload_file(foto, file_path)
    
    cabecilla.foto = file_path
    db.commit()
    return cabecilla

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    print(Fore.YELLOW + f"HTTPException: {exc.detail}" + Style.RESET_ALL)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(Fore.RED + f"Error no manejado: {str(exc)}" + Style.RESET_ALL)
    return JSONResponse(
        status_code=500,
        content={"detail": "Ha ocurrido un error interno"}
    )

@app.on_event("startup")
async def startup_event():
    print(Fore.GREEN + "Servidor iniciado exitosamente." + Style.RESET_ALL)

@app.on_event("shutdown")
async def shutdown_event():
    print(Fore.YELLOW + "Servidor cerrándose..." + Style.RESET_ALL)

def signal_handler(signum, frame):
    print(Fore.YELLOW + "\nDetención solicitada. Cerrando el servidor..." + Style.RESET_ALL)
    asyncio.get_event_loop().stop()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    config = uvicorn.Config(app, host="0.0.0.0", port=9000)
    server = uvicorn.Server(config)
    
    try:
        print(Fore.CYAN + "Iniciando servidor..." + Style.RESET_ALL)
        server.run()
    except Exception as e:
        print(Fore.RED + f"Error al iniciar el servidor: {str(e)}" + Style.RESET_ALL)
    finally:
        print(Fore.CYAN + "Servidor detenido." + Style.RESET_ALL)
