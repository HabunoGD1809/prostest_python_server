# Librerías estándar
import asyncio
import logging
import os
import shutil
import sys
import uuid
import imghdr
from signal import signal
from datetime import date, datetime, timedelta, timezone
from typing import Generic, List, Optional, TypeVar
from contextlib import asynccontextmanager

# Librerías de terceros
from fastapi import (Body, FastAPI, Depends, File, Form, HTTPException, Query, Request, UploadFile, status)
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from fastapi import HTTPException, status
from sqlalchemy import DateTime, create_engine, Column, String, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, joinedload
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.schema import CreateSchema
from sqlalchemy.sql import text
from pydantic import BaseModel, EmailStr, Field, field_validator, validator
import psycopg2
import bcrypt
import uvicorn
from colorama import init, Fore, Style
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

init(autoreset=True)

# Variables de configuración
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Conexión a la base de datos utilizando psycopg2
try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    print(Fore.BLUE + "Conexión exitosa a la base de datos" + Style.RESET_ALL)
    conn.close()
except Exception as e:
    print(Fore.RED + f"Error al conectar a la base de datos: {str(e)}" + Style.RESET_ALL)
    sys.exit(1)

# Configuración de SQLAlchemy
URL_BASE_DE_DATOS = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?options=-c search_path=api"
motor = create_engine(URL_BASE_DE_DATOS)

# el schema 'api' existe y se está utilizando ??
with motor.connect() as conn:
    if not conn.dialect.has_schema(conn, 'api'):
        conn.execute(CreateSchema('api'))
    conn.execute(text('SET search_path TO api'))

SesionLocal = sessionmaker(autocommit=False, autoflush=False, bind=motor)
Base = declarative_base()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código de inicio
    print(Fore.GREEN + "Servidor iniciado exitosamente." + Style.RESET_ALL)
    yield
    # Código de cierre
    print(Fore.YELLOW + "Servidor cerrándose..." + Style.RESET_ALL)

app = FastAPI(lifespan=lifespan)

# Configuracion CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de autenticación
CLAVE_SECRETA = os.getenv("API_SECRET_KEY")
ALGORITMO = os.getenv("API_ALGORITHM")
MINUTOS_EXPIRACION_TOKEN_ACCESO = int(os.getenv("API_ACCESS_TOKEN_EXPIRE_MINUTES"))
MINUTOS_INACTIVIDAD_PERMITIDOS = int(os.getenv("API_INACTIVITY_MINUTES"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Definición del tipo GUID personalizado
class GUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                return "%.32x" % value.int

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value

# Modelos SQLAlchemy
class Usuario(Base):
    __tablename__ = "usuarios"
    __table_args__ = {"schema": "api"}
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    foto = Column(String)
    nombre = Column(String, nullable=False)
    apellidos = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    fecha_creacion = Column(Date, default=date.today())
    soft_delete = Column(Boolean, default=False)
    rol = Column(String, default="usuario", nullable=False)

class Provincia(Base):
    __tablename__ = "provincias"
    __table_args__ = {"schema": "api"}
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    nombre = Column(String, unique=True, nullable=False)
    soft_delete = Column(Boolean, default=False)

class Naturaleza(Base):
    __tablename__ = "naturalezas"
    __table_args__ = {"schema": "api"}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nombre = Column(String, unique=True, nullable=False)
    color = Column(String(7), nullable=False)
    icono = Column(String, nullable=False)
    creado_por = Column(UUID(as_uuid=True), ForeignKey("api.usuarios.id"))
    fecha_creacion = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    soft_delete = Column(Boolean, default=False)

class Cabecilla(Base):
    __tablename__ = "cabecillas"
    __table_args__ = {"schema": "api"}
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    foto = Column(String)
    nombre = Column(String, nullable=False)
    apellido = Column(String, nullable=False)
    cedula = Column(String, unique=True, nullable=False)
    telefono = Column(String)
    direccion = Column(String)
    creado_por = Column(GUID(), ForeignKey("api.usuarios.id"))
    fecha_creacion = Column(Date, default=date.today())
    soft_delete = Column(Boolean, default=False)

class Protesta(Base):
    __tablename__ = "protestas"
    __table_args__ = {"schema": "api"}
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    nombre = Column(String, nullable=False)
    naturaleza_id = Column(GUID(), ForeignKey("api.naturalezas.id"))
    provincia_id = Column(GUID(), ForeignKey("api.provincias.id"))
    resumen = Column(String)
    fecha_evento = Column(Date, nullable=False)
    creado_por = Column(GUID(), ForeignKey("api.usuarios.id"))
    fecha_creacion = Column(Date, default=date.today())
    soft_delete = Column(Boolean, default=False)

    naturaleza = relationship("Naturaleza")
    provincia = relationship("Provincia")
    cabecillas = relationship("Cabecilla", secondary="api.protestas_cabecillas")

class ProtestaCabecilla(Base):
    __tablename__ = "protestas_cabecillas"
    __table_args__ = {"schema": "api"}
    protesta_id = Column(GUID(), ForeignKey("api.protestas.id"), primary_key=True)
    cabecilla_id = Column(GUID(), ForeignKey("api.cabecillas.id"), primary_key=True)

# Modelos Pydantic para la API
class UsuarioSalida(BaseModel):
    id: uuid.UUID
    foto: Optional[str]
    nombre: str
    apellidos: str
    email: EmailStr
    rol: str

    class Config:
        from_attributes = True
class CrearUsuario(BaseModel):
    foto: Optional[str] = None
    nombre: str
    apellidos: str
    email: EmailStr
    password: str
    repetir_password: str

    @field_validator("repetir_password")
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Las contraseñas no coinciden")
        return v
class Token(BaseModel):
    token_acceso: str
    token_actualizacion: str
    tipo_token: str
class CrearNaturaleza(BaseModel):
    nombre: str
    color: str
    icono: Optional[str]
class NaturalezaSalida(BaseModel):
    id: str
    nombre: str
    color: str
    icono: str
    creado_por: str
    fecha_creacion: date
    soft_delete: bool
    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        return cls(
            id=str(obj.id),
            nombre=obj.nombre,
            color=obj.color,
            icono=obj.icono,
            creado_por=str(obj.creado_por),
            fecha_creacion=obj.fecha_creacion.date() if isinstance(obj.fecha_creacion, datetime) else obj.fecha_creacion,
            soft_delete=obj.soft_delete
        )
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
    
    @validator('foto', pre=True)
    def get_full_foto_url(cls, v):
        return get_full_image_url(v)
    class Config:
        from_attributes = True
        json_encoders = {
            uuid.UUID: lambda v: str(v),
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat(),
        }
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
        json_encoders = {
            uuid.UUID: lambda v: str(v),
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat(),
        }
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

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    pages: int
    class Config:
        from_attributes = True

# Funciones Auxiliares de Base de Datos
def obtener_db():
    db = SesionLocal()
    try:
        yield db
    finally:
        db.close()
        
# Funciones de Autenticación y Seguridad
def verificar_password(password_plano: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password_plano.encode("utf-8"), password_hash.encode("utf-8"))

def obtener_hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crear_token_acceso(datos: dict) -> str:
    a_codificar = datos.copy()
    expira = datetime.now(timezone.utc) + timedelta(minutes=MINUTOS_INACTIVIDAD_PERMITIDOS)  # 15 minutos de expiración
    a_codificar.update({
        "exp": expira.timestamp(),
        "ultima_actividad": datetime.now(timezone.utc).isoformat()
    })
    token_jwt_codificado = jwt.encode(a_codificar, CLAVE_SECRETA, algorithm=ALGORITMO)
    return token_jwt_codificado

def crear_token_actualizacion(datos: dict) -> str:
    a_codificar = datos.copy()
    expira = datetime.now(timezone.utc) + timedelta(days=1)  # 1 día de expiración
    a_codificar.update({"exp": expira.timestamp()})
    token_jwt_codificado = jwt.encode(a_codificar, CLAVE_SECRETA, algorithm=ALGORITMO)
    return token_jwt_codificado

async def verificar_token_y_actividad(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, CLAVE_SECRETA, algorithms=[ALGORITMO])
        email: str = payload.get("sub")
        ultima_actividad_str = payload.get("ultima_actividad")

        if email is None or ultima_actividad_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Token inválido"
            )

        ultima_actividad = datetime.fromisoformat(ultima_actividad_str)
        tiempo_inactivo = datetime.now(timezone.utc) - ultima_actividad

        if tiempo_inactivo > timedelta(minutes=MINUTOS_INACTIVIDAD_PERMITIDOS):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Sesión expirada por inactividad"
            )

        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Token expirado"
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Token inválido"
        )

async def obtener_usuario_actual(token: str = Depends(oauth2_scheme), db: Session = Depends(obtener_db)):
    email = await verificar_token_y_actividad(token)
    usuario = db.query(Usuario).filter(Usuario.email == email).first()
    if usuario is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado")
    return usuario

# Funciones de Control de Acceso
def es_admin(usuario: Usuario) -> bool:
    return usuario.rol == "admin"

def verificar_admin(usuario: Usuario = Depends(obtener_usuario_actual)):
    if not es_admin(usuario):
        raise HTTPException(status_code=403, detail="Se requieren permisos de administrador")
    return usuario

# Funciones de Paginación
def paginar(query, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)):
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }

# Middleware
@app.middleware("http")
async def actualizar_token_actividad(request: Request, call_next):
    response = await call_next(request)
    if "Authorization" in request.headers:
        try:
            token = request.headers["Authorization"].split()[1]
            payload = jwt.decode(token, CLAVE_SECRETA, algorithms=[ALGORITMO])
            nuevo_token = crear_token_acceso({"sub": payload["sub"]})
            response.headers["New-Token"] = nuevo_token
        except JWTError:
            # Si hay una excepción, no actualizamos el token
            pass
    return response

UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE"))
ALLOWED_IMAGE_TYPES = os.getenv("ALLOWED_IMAGE_TYPES").split(',')

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Rutas de la API
@app.put("/usuarios/{usuario_id}/rol", response_model=UsuarioSalida)
def cambiar_rol_usuario(
    usuario_id: uuid.UUID,
    nuevo_rol: str = Query(..., regex="^(admin|usuario)$"),
    admin_actual: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        # Verificar si el admin está intentando cambiar su propio rol
        if usuario.id == admin_actual.id:
            raise HTTPException(status_code=422, detail="No puedes cambiar tu propio rol de administrador")
        
        # Verificar si se está intentando cambiar el rol del último administrador
        if usuario.rol == 'admin' and nuevo_rol == 'usuario':
            admin_count = db.query(Usuario).filter(Usuario.rol == 'admin', Usuario.soft_delete == False).count()
            if admin_count == 1:
                raise HTTPException(status_code=422, detail="No se puede cambiar el rol del último administrador")

        # Verificar si el rol ya es el que se está intentando asignar
        if usuario.rol == nuevo_rol:
            return usuario  # Retornar el usuario sin cambios si el rol es el mismo

        # Permitir cambiar el rol de otros usuarios
        usuario.rol = nuevo_rol
        db.commit()
        db.refresh(usuario)
        print(Fore.GREEN + f"Rol de usuario actualizado exitosamente: {usuario.email} - Nuevo rol: {nuevo_rol}" + Style.RESET_ALL)
        return usuario
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al cambiar rol de usuario: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/usuarios", response_model=List[UsuarioSalida])
def listar_usuarios(
    admin_actual: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        usuarios = db.query(Usuario).filter(Usuario.soft_delete == False).all()
        return usuarios
    except Exception as e:
        print(Fore.RED + f"Error al listar usuarios: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    
@app.get("/usuarios/me", response_model=UsuarioSalida)
async def obtener_usuario_actual_ruta(usuario: Usuario = Depends(obtener_usuario_actual)):
    return UsuarioSalida.model_validate(usuario)

def verificar_autenticacion(usuario: Usuario = Depends(obtener_usuario_actual)):
    if not usuario:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")
    return usuario

# nuevas rutas admin
@app.post("/admin/usuarios", response_model=UsuarioSalida)
def crear_usuario_admin(
    usuario: CrearUsuario,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        hash_password = obtener_hash_password(usuario.password)
        db_usuario = Usuario(
            foto=usuario.foto,
            nombre=usuario.nombre,
            apellidos=usuario.apellidos,
            email=usuario.email,
            password=hash_password,
            rol=usuario.rol if usuario.rol else "usuario"
        )
        db.add(db_usuario)
        db.commit()
        db.refresh(db_usuario)
        print(Fore.GREEN + f"Usuario creado exitosamente por admin: {usuario.email}" + Style.RESET_ALL)
        return db_usuario
    except IntegrityError as e:
        db.rollback()
        if "usuarios_email_key" in str(e.orig):
            raise HTTPException(status_code=400, detail=f"Ya existe un usuario con el email '{usuario.email}'")
        raise HTTPException(status_code=400, detail="Error al crear el usuario")
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al crear usuario: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/admin/usuarios/{usuario_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_usuario_admin(
    usuario_id: uuid.UUID,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        db_usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not db_usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        db_usuario.soft_delete = True
        db.commit()
        print(Fore.GREEN + f"Usuario eliminado exitosamente por admin: {db_usuario.email}" + Style.RESET_ALL)
        return {"detail": "Usuario eliminado exitosamente"}
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al eliminar usuario: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/registro", response_model=UsuarioSalida)
async def registrar_usuario(
    nombre: str = Form(...),
    apellidos: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    repetir_password: str = Form(...),
    foto: UploadFile = File(None),
    db: Session = Depends(obtener_db),
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
            password=hash_password,
        )
        db.add(db_usuario)
        db.commit()
        db.refresh(db_usuario)
        print(
            Fore.GREEN + f"Usuario registrado exitosamente: {email}" + Style.RESET_ALL
        )
        return db_usuario
    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "usuarios_email_key" in str(e.orig):
                print(
                    Fore.YELLOW
                    + f"Intento de registrar usuario con email duplicado: {email}"
                    + Style.RESET_ALL
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Ya existe un usuario con el email '{email}'",
                )
        print(
            Fore.RED
            + f"Error de integridad al registrar usuario: {str(e)}"
            + Style.RESET_ALL
        )
        raise HTTPException(
            status_code=400,
            detail="Error al registrar el usuario. Por favor, intente de nuevo.",
        )
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al registrar usuario: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/token", response_model=Token)
async def login_para_token_acceso(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(obtener_db)
):
    logger.info(f"Intento de inicio de sesión para: {form_data.username}")
    usuario = db.query(Usuario).filter(Usuario.email == form_data.username).first()
    if not usuario or not verificar_password(form_data.password, usuario.password):
        logger.warning(f"Intento de inicio de sesión fallido para: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_acceso = crear_token_acceso({"sub": usuario.email})
    token_actualizacion = crear_token_actualizacion({"sub": usuario.email})
    logger.info(f"Inicio de sesión exitoso para: {usuario.email}")
    logger.debug(f"Token de acceso generado: {token_acceso}")
    logger.debug(f"Token de actualización generado: {token_actualizacion}")
    return Token(
        token_acceso=token_acceso,
        token_actualizacion=token_actualizacion,
        tipo_token="bearer"
    )

@app.post("/token/renovar", response_model=Token)
async def renovar_token(
    token_actualizacion: str = Body(..., embed=True),
    db: Session = Depends(obtener_db)
):
    try:
        logger.info("Intento de renovación de token")
        logger.debug(f"Token de actualización recibido: {token_actualizacion}")
        
        if not token_actualizacion or not isinstance(token_actualizacion, str):
            logger.error(f"Token de actualización inválido: {token_actualizacion}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token de actualización inválido")
        
        try:
            payload = jwt.decode(token_actualizacion, CLAVE_SECRETA, algorithms=[ALGORITMO])
            logger.debug(f"Payload decodificado: {payload}")
        except jwt.JWTError as e:
            logger.error(f"Error al decodificar el token: {str(e)}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token inválido: {str(e)}")
        
        email: str = payload.get("sub")
        exp: float = payload.get("exp")
        
        if email is None or exp is None:
            logger.warning(f"Payload del token inválido: {payload}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token de actualización inválido")
        
        # Verificar si el token ha expirado
        if datetime.now(timezone.utc).timestamp() > exp:
            logger.warning(f"Token de actualización expirado para: {email}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token de actualización expirado")
        
        usuario = db.query(Usuario).filter(Usuario.email == email).first()
        if usuario is None:
            logger.warning(f"Usuario no encontrado para el email: {email}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado")
        
        nuevo_token_acceso = crear_token_acceso({"sub": email})
        logger.debug(f"Nuevo token de acceso generado: {nuevo_token_acceso}")
        
        # Calcular el tiempo restante para la expiración del token de actualización
        tiempo_expiracion = exp - datetime.now(timezone.utc).timestamp()
        if tiempo_expiracion < 3600:  # Si falta menos de una hora para que expire
            logger.info(f"Creando nuevo token de actualización para: {email}")
            nuevo_token_actualizacion = crear_token_actualizacion({"sub": email})
        else:
            nuevo_token_actualizacion = token_actualizacion
        
        logger.info(f"Token renovado exitosamente para: {email}")
        return Token(
            token_acceso=nuevo_token_acceso,
            token_actualizacion=nuevo_token_actualizacion,
            tipo_token="bearer"
        )
    except Exception as e:
        logger.exception(f"Error inesperado durante la renovación del token: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error interno del servidor: {str(e)}")

@app.get("/pagina-principal", response_model=ResumenPrincipal)
def obtener_resumen_principal(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    total_protestas = db.query(Protesta).filter(Protesta.soft_delete == False).count()
    protestas_recientes = (
        db.query(Protesta)
        .filter(Protesta.soft_delete == False)
        .order_by(Protesta.fecha_creacion.desc())
        .limit(5)
        .all()
    )

    return ResumenPrincipal(
        total_protestas=total_protestas, protestas_recientes=protestas_recientes
    )

@app.post("/naturalezas", response_model=NaturalezaSalida)
def crear_naturaleza(
    naturaleza: CrearNaturaleza,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        db_naturaleza = Naturaleza(
            **naturaleza.model_dump(), creado_por=usuario.id
        )
        db.add(db_naturaleza)
        db.commit()
        db.refresh(db_naturaleza)
        print(
            Fore.GREEN
            + f"Naturaleza creada exitosamente: {naturaleza.nombre}"
            + Style.RESET_ALL
        )
        return NaturalezaSalida.from_orm(db_naturaleza)
    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "naturalezas_nombre_key" in str(e.orig):
                print(
                    Fore.YELLOW
                    + f"Intento de crear naturaleza con nombre duplicado: {naturaleza.nombre}"
                    + Style.RESET_ALL
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Ya existe una naturaleza con el nombre '{naturaleza.nombre}'",
                )
        print(
            Fore.RED
            + f"Error de integridad al crear naturaleza: {str(e)}"
            + Style.RESET_ALL
        )
        raise HTTPException(
            status_code=400,
            detail="Error al crear la naturaleza. Por favor, intente de nuevo.",
        )
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al crear naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/cabecillas", response_model=CabecillaSalida)
async def crear_cabecilla(
    nombre: str = Form(...),
    apellido: str = Form(...),
    cedula: str = Form(...),
    telefono: str = Form(None),
    direccion: str = Form(None),
    foto: UploadFile = File(None),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        # Crear el objeto Cabecilla sin la foto primero
        cabecilla_data = {
            "nombre": nombre,
            "apellido": apellido,
            "cedula": cedula,
            "telefono": telefono,
            "direccion": direccion,
            "creado_por": usuario.id,
        }
        db_cabecilla = Cabecilla(**cabecilla_data)
        db.add(db_cabecilla)
        db.flush()  # Para obtener el ID generado

        # Manejar la subida de la foto si se proporciona
        if foto:
            file_extension = os.path.splitext(foto.filename)[1]
            file_name = f"cabecilla_{db_cabecilla.id}{file_extension}"
            relative_path = os.path.join(UPLOAD_DIR, file_name)
            file_path = os.path.join(STATIC_FILES_DIR, relative_path)

            if save_upload_file(foto, file_path):
                db_cabecilla.foto = relative_path
            else:
                db.rollback()
                raise HTTPException(status_code=500, detail="Error al guardar la imagen")

        db.commit()
        db.refresh(db_cabecilla)
        return db_cabecilla

    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "cabecillas_cedula_key" in str(e.orig):
                print(
                    Fore.YELLOW
                    + f"Intento de crear cabecilla con cédula duplicada: {cedula}"
                    + Style.RESET_ALL
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Ya existe un cabecilla con la cédula '{cedula}'",
                )
        print(
            Fore.RED
            + f"Error de integridad al crear cabecilla: {str(e)}"
            + Style.RESET_ALL
        )
        raise HTTPException(
            status_code=400,
            detail="Error al crear el cabecilla. Por favor, intente de nuevo.",
        )
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al crear cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/protestas/completa", response_model=ProtestaSalida)
def crear_protesta_completa(
    protesta: CrearProtestaCompleta,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    # Crear nueva naturaleza si se proporciona
    if protesta.nueva_naturaleza:
        nueva_naturaleza = Naturaleza(
            **protesta.nueva_naturaleza.model_dump(), creado_por=usuario.id
        )
        db.add(nueva_naturaleza)
        db.flush()
        naturaleza_id = nueva_naturaleza.id
    else:
        naturaleza_id = protesta.naturaleza_id

    # Crear nuevos cabecillas
    nuevos_cabecillas_ids = []
    for nuevo_cabecilla in protesta.nuevos_cabecillas:
        db_cabecilla = Cabecilla(
            **nuevo_cabecilla.model_dump(), creado_por=usuario.id
        )
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
        creado_por=usuario.id,
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
def crear_protesta(
    protesta: CrearProtesta,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        # Validar que la naturaleza y la provincia existen
        naturaleza = db.query(Naturaleza).get(protesta.naturaleza_id)
        provincia = db.query(Provincia).get(protesta.provincia_id)
        if not naturaleza or not provincia:
            raise HTTPException(
                status_code=400, detail="Naturaleza o provincia no válida"
            )

        # Validar que los cabecillas existen
        cabecillas = (
            db.query(Cabecilla).filter(Cabecilla.id.in_(protesta.cabecillas)).all()
        )
        if len(cabecillas) != len(protesta.cabecillas):
            raise HTTPException(
                status_code=400, detail="Uno o más cabecillas no son válidos"
            )

        db_protesta = Protesta(
            nombre=protesta.nombre,
            naturaleza_id=protesta.naturaleza_id,
            provincia_id=protesta.provincia_id,
            resumen=protesta.resumen,
            fecha_evento=protesta.fecha_evento,
            creado_por=usuario.id,
        )
        db.add(db_protesta)
        db.flush()

        for cabecilla_id in protesta.cabecillas:
            db_protesta.cabecillas.append(db.query(Cabecilla).get(cabecilla_id))

        db.commit()
        db.refresh(db_protesta)
        print(
            Fore.GREEN
            + f"Protesta creada exitosamente: {protesta.nombre}"
            + Style.RESET_ALL
        )
        return db_protesta
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al crear protesta: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(
            status_code=500, detail=f"Error interno del servidor: {str(e)}"
        )

@app.get("/protestas", response_model=PaginatedResponse[ProtestaSalida])
def obtener_protestas(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    fecha_desde: Optional[date] = None,
    fecha_hasta: Optional[date] = None,
    provincia_id: Optional[uuid.UUID] = None,
    naturaleza_id: Optional[uuid.UUID] = None,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
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
    pages = (total + page_size - 1) // page_size

    protestas = query.order_by(Protesta.fecha_evento.desc())
    protestas = protestas.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "items": protestas,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }

@app.get("/protestas/{protesta_id}", response_model=ProtestaSalida)
def obtener_protesta(
    protesta_id: str,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        # Convertir el string a UUID
        protesta_uuid = uuid.UUID(protesta_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de protesta inválido")

    try:
        protesta = (
            db.query(Protesta)
            .options(
                joinedload(Protesta.naturaleza),
                joinedload(Protesta.provincia),
                joinedload(Protesta.cabecillas),
            )
            .filter(Protesta.id == protesta_uuid, Protesta.soft_delete == False)
            .first()
        )

        if not protesta:
            raise HTTPException(status_code=404, detail="Protesta no encontrada")

        # Convertir fechas a objetos date si son datetime
        if isinstance(protesta.fecha_evento, datetime):
            protesta.fecha_evento = protesta.fecha_evento.date()
        if isinstance(protesta.fecha_creacion, datetime):
            protesta.fecha_creacion = protesta.fecha_creacion.date()

        # Crear el diccionario de la protesta
        protesta_dict = {
            "id": protesta.id,
            "nombre": protesta.nombre,
            "naturaleza_id": protesta.naturaleza_id,
            "provincia_id": protesta.provincia_id,
            "resumen": protesta.resumen,
            "fecha_evento": protesta.fecha_evento,
            "creado_por": protesta.creado_por,
            "fecha_creacion": protesta.fecha_creacion,
            "soft_delete": protesta.soft_delete,
            "cabecillas": [
                CabecillaSalida.model_validate(c) for c in protesta.cabecillas
            ],
        }

        return ProtestaSalida(**protesta_dict)
    except Exception as e:
        print(f"Error al obtener protesta: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error interno del servidor: {str(e)}"
        )

@app.put("/protestas/{protesta_id}", response_model=ProtestaSalida)
def actualizar_protesta(
    protesta_id: uuid.UUID,
    protesta: CrearProtesta,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id).first()
        if not db_protesta:
            raise HTTPException(status_code=404, detail="Protesta no encontrada")

        # Verificar si el usuario es el creador de la protesta
        if db_protesta.creado_por != usuario.id:
            raise HTTPException(status_code=403, detail="No tienes permiso para editar esta protesta")

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
def eliminar_protesta(
    protesta_id: uuid.UUID,
    admin_actual: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id).first()
        if not db_protesta:
            raise HTTPException(status_code=404, detail="Protesta no encontrada")
        
        db_protesta.soft_delete = True
        db.commit()
        print(Fore.GREEN + f"Protesta eliminada exitosamente por admin: {db_protesta.nombre}" + Style.RESET_ALL)
        return {"detail": "Protesta eliminada exitosamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al eliminar protesta: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/provincias", response_model=List[ProvinciaSalida])
def obtener_provincias(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        provincias = db.query(Provincia).filter(Provincia.soft_delete == False).all()
        print(
            Fore.GREEN
            + f"Provincias obtenidas exitosamente. Total: {len(provincias)}"
            + Style.RESET_ALL
        )
        return provincias
    except Exception as e:
        print(Fore.RED + f"Error al obtener provincias: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/provincias/{provincia_id}", response_model=ProvinciaSalida)
def obtener_provincia(
    provincia_id: uuid.UUID,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        provincia = (
            db.query(Provincia)
            .filter(Provincia.id == provincia_id, Provincia.soft_delete == False)
            .first()
        )
        if not provincia:
            raise HTTPException(status_code=404, detail="Provincia no encontrada")
        return provincia
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/naturalezas", response_model=PaginatedResponse[NaturalezaSalida])
def obtener_naturalezas(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    nombre: Optional[str] = None,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        query = db.query(Naturaleza).filter(Naturaleza.soft_delete == False)
        
        if nombre:
            query = query.filter(Naturaleza.nombre.ilike(f"%{nombre}%"))
        
        total = query.count()
        naturalezas = query.offset((page - 1) * page_size).limit(page_size).all()
        naturalezas_salida = [NaturalezaSalida.from_orm(n) for n in naturalezas]
        
        result = {
            "items": naturalezas_salida,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size
        }
        
        print(
            Fore.GREEN
            + f"Naturalezas obtenidas exitosamente. Total: {total}, Página: {page}"
            + Style.RESET_ALL
        )
        return result
    except Exception as e:
        print(Fore.RED + f"Error al obtener naturalezas: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.get("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def obtener_naturaleza(
    naturaleza_id: uuid.UUID,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        naturaleza = (
            db.query(Naturaleza)
            .filter(Naturaleza.id == naturaleza_id, Naturaleza.soft_delete == False)
            .first()
        )
        if not naturaleza:
            print(
                Fore.YELLOW
                + f"Naturaleza no encontrada: {naturaleza_id}"
                + Style.RESET_ALL
            )
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")
        print(
            Fore.GREEN
            + f"Naturaleza obtenida exitosamente: {naturaleza.nombre}"
            + Style.RESET_ALL
        )
        return NaturalezaSalida.from_orm(naturaleza)
    except HTTPException as he:
        raise he
    except Exception as e:
        print(Fore.RED + f"Error al obtener naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def actualizar_naturaleza(
    naturaleza_id: uuid.UUID,
    naturaleza: CrearNaturaleza,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        db_naturaleza = (
            db.query(Naturaleza)
            .filter(
                Naturaleza.id == naturaleza_id,
                Naturaleza.creado_por == usuario.id,
            )
            .first()
        )
        if not db_naturaleza:
            print(
                Fore.YELLOW
                + f"Naturaleza no encontrada o sin permisos: {naturaleza_id}"
                + Style.RESET_ALL
            )
            raise HTTPException(
                status_code=404,
                detail="Naturaleza no encontrada o no tienes permiso para editarla",
            )

        for key, value in naturaleza.model_dump().items():
            setattr(db_naturaleza, key, value)

        db.commit()
        db.refresh(db_naturaleza)
        print(
            Fore.GREEN
            + f"Naturaleza actualizada exitosamente: {db_naturaleza.nombre}"
            + Style.RESET_ALL
        )
        return NaturalezaSalida.from_orm(db_naturaleza)
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al actualizar naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/naturalezas/{naturaleza_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_naturaleza(
    naturaleza_id: uuid.UUID,
    admin: Usuario = Depends(verificar_admin),  # Cambiado a verificar_admin
    db: Session = Depends(obtener_db),
):
    try:
        db_naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id).first()
        if not db_naturaleza:
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")

        # Verificar si la naturaleza está asociada a alguna protesta
        protestas_asociadas = db.query(Protesta).filter(Protesta.naturaleza_id == naturaleza_id).first()
        if protestas_asociadas:
            raise HTTPException(
                status_code=400,
                detail="No se puede eliminar una naturaleza asociada a protestas. Elimine o edite las protestas primero."
            )

        db_naturaleza.soft_delete = True
        db.commit()
        print(Fore.GREEN + f"Naturaleza eliminada exitosamente por admin: {db_naturaleza.nombre}" + Style.RESET_ALL)
        return {"detail": "Naturaleza eliminada exitosamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al eliminar naturaleza: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/cabecillas", response_model=PaginatedResponse[CabecillaSalida])
def obtener_cabecillas(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    nombre: Optional[str] = None,
    apellido: Optional[str] = None,
    cedula: Optional[str] = None,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        query = db.query(Cabecilla).filter(Cabecilla.soft_delete == False)
        
        if nombre:
            query = query.filter(Cabecilla.nombre.ilike(f"%{nombre}%"))
        if apellido:
            query = query.filter(Cabecilla.apellido.ilike(f"%{apellido}%"))
        if cedula:
            query = query.filter(Cabecilla.cedula.ilike(f"%{cedula}%"))
        
        total = query.count()
        cabecillas = query.offset((page - 1) * page_size).limit(page_size).all()
        
        result = {
            "items": cabecillas,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size
        }
        
        print(
            Fore.GREEN
            + f"Cabecillas obtenidos exitosamente. Total: {total}, Página: {page}"
            + Style.RESET_ALL
        )
        return result
    except Exception as e:
        print(Fore.RED + f"Error al obtener cabecillas: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
def obtener_cabecilla(
    cabecilla_id: uuid.UUID,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        cabecilla = (
            db.query(Cabecilla)
            .filter(Cabecilla.id == cabecilla_id, Cabecilla.soft_delete == False)
            .first()
        )
        if not cabecilla:
            print(
                Fore.YELLOW
                + f"Cabecilla no encontrado: {cabecilla_id}"
                + Style.RESET_ALL
            )
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")
        print(
            Fore.GREEN
            + f"Cabecilla obtenido exitosamente: {cabecilla.nombre} {cabecilla.apellido}"
            + Style.RESET_ALL
        )
        return cabecilla
    except HTTPException as he:
        raise he
    except Exception as e:
        print(Fore.RED + f"Error al obtener cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
def actualizar_cabecilla(
    cabecilla_id: uuid.UUID,
    cabecilla: CrearCabecilla,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        db_cabecilla = (
            db.query(Cabecilla)
            .filter(
                Cabecilla.id == cabecilla_id, Cabecilla.creado_por == usuario.id
            )
            .first()
        )
        if not db_cabecilla:
            print(
                Fore.YELLOW
                + f"Cabecilla no encontrado o sin permisos: {cabecilla_id}"
                + Style.RESET_ALL
            )
            raise HTTPException(
                status_code=404,
                detail="Cabecilla no encontrado o no tienes permiso para editarlo",
            )

        for key, value in cabecilla.model_dump().items():
            setattr(db_cabecilla, key, value)

        db.commit()
        db.refresh(db_cabecilla)
        print(
            Fore.GREEN
            + f"Cabecilla actualizado exitosamente: {db_cabecilla.nombre} {db_cabecilla.apellido}"
            + Style.RESET_ALL
        )
        return db_cabecilla
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al actualizar cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/cabecillas/{cabecilla_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_cabecilla(
    cabecilla_id: uuid.UUID,
    usuario: Usuario = Depends(verificar_admin),  # Cambiado a verificar_admin
    db: Session = Depends(obtener_db),
):
    try:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id).first()
        if not db_cabecilla:
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")

        # Eliminar asociaciones con protestas
        db.query(ProtestaCabecilla).filter(ProtestaCabecilla.cabecilla_id == cabecilla_id).delete()

        db_cabecilla.soft_delete = True
        db.commit()
        print(Fore.GREEN + f"Cabecilla eliminado exitosamente por admin: {db_cabecilla.nombre} {db_cabecilla.apellido}" + Style.RESET_ALL)
        return {"detail": "Cabecilla eliminado exitosamente"}
    except Exception as e:
        db.rollback()
        print(Fore.RED + f"Error al eliminar cabecilla: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# NUEVO ENDPOINT
@app.get("/cabecillas/all", response_model=List[CabecillaSalida])
def obtener_todos_los_cabecillas(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        cabecillas = db.query(Cabecilla).filter(Cabecilla.soft_delete == False).all()
        return [CabecillaSalida.model_validate(c) for c in cabecillas]
    except Exception as e:
        print(Fore.RED + f"Error al obtener todos los cabecillas: {str(e)}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

def validate_image(file: UploadFile):
    # Verificar el tamaño del archivo
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo es demasiado grande. El tamaño máximo es de {MAX_IMAGE_SIZE // (1024 * 1024)} MB.",
        )

    # Verificar el tipo de archivo
    contents = file.file.read()
    file.file.seek(0)
    file_type = imghdr.what(None, h=contents)
    if file_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Solo se aceptan {', '.join(ALLOWED_IMAGE_TYPES)}.",
        )

# Configurar la ruta base para los archivos estáticos
STATIC_FILES_DIR = "static"
UPLOAD_DIR = "uploads"
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

# Función auxiliar para guardar el archivo
def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    try:
        with open(destination, "wb") as buffer:
            content = upload_file.file.read()
            buffer.write(content)
        return destination
    except Exception as e:
        print(f"Error al guardar el archivo: {str(e)}")
        return None

@app.post("/usuarios/foto", response_model=UsuarioSalida)
async def actualizar_foto_usuario(
    foto: UploadFile = File(...),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    validate_image(foto)
    file_name = f"{usuario.id}_{foto.filename}"
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    save_upload_file(foto, file_path)

    usuario.foto = file_path
    db.commit()
    return usuario

# Modificar la función para obtener la URL completa de la foto
def get_full_image_url(foto_path: str) -> str:
    if foto_path:
        return f"/static/{foto_path}"
    return None

@app.post("/cabecillas/{cabecilla_id}/foto", response_model=CabecillaSalida)
async def actualizar_foto_cabecilla(
    cabecilla_id: uuid.UUID,
    foto: UploadFile = File(...),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.creado_por == usuario.id).first()
    if not cabecilla:
        raise HTTPException(status_code=404, detail="Cabecilla no encontrado o no tienes permiso para editarlo")

    validate_image(foto)
    file_extension = os.path.splitext(foto.filename)[1]
    file_name = f"cabecilla_{cabecilla_id}{file_extension}"
    relative_path = os.path.join(UPLOAD_DIR, file_name)
    file_path = os.path.join(STATIC_FILES_DIR, relative_path)
    
    if save_upload_file(foto, file_path):
        cabecilla.foto = relative_path
        db.commit()
        return cabecilla
    else:
        raise HTTPException(status_code=500, detail="Error al guardar la imagen")

# Manejadores de excepciones
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    print(Fore.YELLOW + f"HTTPException: {exc.detail}" + Style.RESET_ALL)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(Fore.RED + f"Error no manejado: {str(exc)}" + Style.RESET_ALL)
    return JSONResponse(
        status_code=500, content={"detail": "Ha ocurrido un error interno"}
    )

# Variable global para controlar el ciclo del servidor
server_should_exit = False

def signal_handler(signum, frame):
    global server_should_exit
    print(
        Fore.YELLOW
        + "\nDetención solicitada. Cerrando el servidor..."
        + Style.RESET_ALL
    )
    server_should_exit = True

# Configuración del servidor
config = uvicorn.Config(
    app,
    host=os.getenv("SERVER_HOST"),
    port=int(os.getenv("SERVER_PORT"))
)
server = uvicorn.Server(config)

# Punto de entrada 
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # Manejo señal de salida
    async def run_server():
        await server.serve()
        while not server_should_exit:
            await asyncio.sleep(1)
        await server.shutdown()

    try:
        print(Fore.CYAN + "Iniciando servidor..." + Style.RESET_ALL)
        asyncio.run(run_server())
    except Exception as e:
        print(Fore.RED + f"Error al iniciar el servidor: {str(e)}" + Style.RESET_ALL)
    finally:
        print(Fore.CYAN + "Servidor detenido." + Style.RESET_ALL)
