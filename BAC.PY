# Librerías estándar
import asyncio
import logging
import os
import sys
import uuid
from PIL import Image
import io
import signal as signal_module
from datetime import date, datetime, timedelta, timezone
from typing import Generic, List, Optional, Tuple, TypeVar, Dict
from contextlib import asynccontextmanager

# Librerías de terceros
from fastapi import (Body, FastAPI, Depends, File, Form, HTTPException, Query, Request, UploadFile, status)
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, FieldValidationInfo
import bcrypt
import uvicorn
from colorama import init, Fore, Style
from dotenv import load_dotenv
from passlib.context import CryptContext

# SQLAlchemy
from sqlalchemy import (DateTime, create_engine, Column, String, Boolean, Date, ForeignKey, func, and_)
from sqlalchemy.orm import sessionmaker, relationship, Session, joinedload, declarative_base, contains_eager
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.schema import CreateSchema
from sqlalchemy.sql import text

# PostgreSQL
import psycopg2

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

# Verificar y crear el schema 'api' si no existe
with motor.connect() as conn:
    if not conn.dialect.has_schema(conn, 'api'):
        conn.execute(CreateSchema('api'))
    conn.execute(text('SET search_path TO api'))

SesionLocal = sessionmaker(autocommit=False, autoflush=False, bind=motor)
Base = declarative_base()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(Fore.GREEN + "Servidor iniciado exitosamente." + Style.RESET_ALL)
    yield
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
    fecha_creacion = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
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
    fecha_creacion = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
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
    creado_por = Column(GUID(), ForeignKey("api.usuarios.id"))
    creador = relationship("Usuario", foreign_keys=[creado_por])

    naturaleza = relationship("Naturaleza")
    provincia = relationship("Provincia")
    cabecillas = relationship("Cabecilla", secondary="api.protestas_cabecillas")

class ProtestaCabecilla(Base):
    __tablename__ = "protestas_cabecillas"
    __table_args__ = {"schema": "api"}
    protesta_id = Column(GUID(), ForeignKey("api.protestas.id"), primary_key=True)
    cabecilla_id = Column(GUID(), ForeignKey("api.cabecillas.id"), primary_key=True)

# Modelos Pydantic
class UsuarioBase(BaseModel):
    nombre: str
    apellidos: str
    email: EmailStr
    rol: str = Field(default="usuario")

    @field_validator("rol")
    def validate_rol(cls, v):
        if v not in ["usuario", "admin"]:
            raise ValueError("El rol debe ser 'usuario' o 'admin'")
        return v

class UsuarioSalida(UsuarioBase):
    id: uuid.UUID
    fecha_creacion: datetime
    foto: Optional[str] = None

    @field_validator("foto", mode="before")
    def get_full_foto_url(cls, v):
        return get_full_image_url(v)

    model_config = ConfigDict(from_attributes=True)

class CrearUsuario(UsuarioBase):
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
    icono: str

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
            fecha_creacion=(
                obj.fecha_creacion.date()
                if isinstance(obj.fecha_creacion, datetime)
                else obj.fecha_creacion
            ),
            soft_delete=obj.soft_delete,
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
    foto: Optional[str] = None
    nombre: str
    apellido: str
    cedula: str
    telefono: Optional[str]
    direccion: Optional[str]
    creado_por: uuid.UUID
    fecha_creacion: date
    soft_delete: bool

    @field_validator("foto", mode="before")
    def get_full_foto_url(cls, v):
        return get_full_image_url(v)

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            uuid.UUID: str,
            date: date.isoformat,
            datetime: datetime.isoformat,
        }
    )

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
    creador_nombre: Optional[str] = None
    creador_email: Optional[str] = None

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

class AllData(BaseModel):
    protestas: PaginatedResponse
    naturalezas: List[NaturalezaSalida]
    provincias: List[ProvinciaSalida]
    cabecillas: List[CabecillaSalida]

    class Config:
        from_attributes = True
        
class VersionResponse(BaseModel):
    version: str

# Nuevas for cambio de contraseña
def validar_contrasena(v: str) -> str:
    if len(v) < 8:
        raise ValueError('La contraseña debe tener al menos 8 caracteres')
    if not any(char.isupper() for char in v):
        raise ValueError('La contraseña debe contener al menos una letra mayúscula')
    if not any(char.islower() for char in v):
        raise ValueError('La contraseña debe contener al menos una letra minúscula')
    if not any(char.isdigit() for char in v):
        raise ValueError('La contraseña debe contener al menos un número')
    return v

class CambioContrasenaUsuario(BaseModel):
    contrasena_actual: str
    nueva_contrasena: str
    confirmar_contrasena: str

    @field_validator('nueva_contrasena')
    def validar_nueva_contrasena(cls, v):
        return validar_contrasena(v)

    @field_validator('confirmar_contrasena')
    def passwords_match(cls, v, info: FieldValidationInfo):
        if 'nueva_contrasena' in info.data and v != info.data['nueva_contrasena']:
            raise ValueError('Las contraseñas no coinciden')
        return v

class RestablecerContrasenaAdmin(BaseModel):
    nueva_contrasena: str

    @field_validator('nueva_contrasena')
    def validar_nueva_contrasena(cls, v):
        return validar_contrasena(v)

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

# Creacion y actualizacion de token
def crear_token_acceso(datos: dict) -> str:
    a_codificar = datos.copy()
    expira = datetime.now(timezone.utc) + timedelta(minutes=MINUTOS_INACTIVIDAD_PERMITIDOS)
    a_codificar.update({"exp": expira.timestamp(), "ultima_actividad": datetime.now(timezone.utc).isoformat()})
    return jwt.encode(a_codificar, CLAVE_SECRETA, algorithm=ALGORITMO)

def crear_token_actualizacion(datos: dict) -> str:
    a_codificar = datos.copy()
    expira = datetime.now(timezone.utc) + timedelta(days=7)
    a_codificar.update({"exp": expira.timestamp()})
    return jwt.encode(a_codificar, CLAVE_SECRETA, algorithm=ALGORITMO)

async def verificar_token_y_actividad(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, CLAVE_SECRETA, algorithms=[ALGORITMO])
        email: str = payload.get("sub")
        ultima_actividad_str = payload.get("ultima_actividad")

        if email is None or ultima_actividad_str is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

        ultima_actividad = datetime.fromisoformat(ultima_actividad_str)
        tiempo_inactivo = datetime.now(timezone.utc) - ultima_actividad

        if tiempo_inactivo > timedelta(minutes=MINUTOS_INACTIVIDAD_PERMITIDOS):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sesión expirada por inactividad")

        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

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

def verificar_propiedad(entidad, usuario: Usuario):
    if not es_admin(usuario) and entidad.creado_por != usuario.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para modificar este recurso")

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

# Dependencia para crear usuario desde formulario
async def crear_usuario_form(
    nombre: str = Form(...),
    apellidos: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    repetir_password: str = Form(...),
    rol: str = Form("usuario"),
    foto: Optional[UploadFile] = File(None),
):
    return (
        CrearUsuario(
            nombre=nombre,
            apellidos=apellidos,
            email=email,
            password=password,
            repetir_password=repetir_password,
            rol=rol,
        ),
        foto,
    )

# Función común para crear usuarios
async def crear_usuario(usuario: CrearUsuario, foto: Optional[UploadFile], db: Session, is_admin_creation: bool = False):
    try:
        hash_password = obtener_hash_password(usuario.password)
        foto_path = None
        if foto:
            validate_image(foto)
            file_name = f"usuario_{uuid.uuid4()}_{foto.filename}"
            file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
            await save_upload_file(foto, file_path)
            foto_path = os.path.join(UPLOAD_DIR, file_name)

        db_usuario = Usuario(
            foto=foto_path,
            nombre=usuario.nombre,
            apellidos=usuario.apellidos,
            email=usuario.email,
            password=hash_password,
            rol=usuario.rol if is_admin_creation else "usuario",
        )
        db.add(db_usuario)
        db.commit()
        db.refresh(db_usuario)
        print(f"Usuario {'creado por admin' if is_admin_creation else 'registrado'} exitosamente: {usuario.email}")
        return db_usuario
    except IntegrityError as e:
        db.rollback()
        if "usuarios_email_key" in str(e.orig):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ya existe un usuario con el email '{usuario.email}'")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error al crear el usuario. Por favor, intente de nuevo.")
    except Exception as e:
        db.rollback()
        print(f"Error al {'crear' if is_admin_creation else 'registrar'} usuario: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")

# Funcion para la version de la aplicacion 
def get_current_version():
    version = os.getenv("APP_VERSION", "1.0.0")
    if not version:
        logger.warning("APP_VERSION no está definida en las variables de entorno. Usando versión por defecto.")
    return version

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
            pass
    return response

# Rutas
@app.get("/api/version", response_model=VersionResponse)
async def get_version():
    try:
        current_version = get_current_version()
        logger.info(f"Versión solicitada: {current_version}")
        return VersionResponse(version=current_version)
    except Exception as e:
        logger.error(f"Error al obtener la versión: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al obtener la versión")

@app.get("/check-user")
def check_user_exists(email: str = Query(..., description="Email del usuario a verificar"), db: Session = Depends(obtener_db)):
    usuario = db.query(Usuario).filter(Usuario.email == email, Usuario.soft_delete == False).first()
    return {"exists": usuario is not None}

@app.get("/pagina-principal", response_model=Dict)
def obtener_resumen_principal(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
    fecha_inicio: Optional[date] = Query(None, description="Fecha de inicio para el reporte"),
    fecha_fin: Optional[date] = Query(None, description="Fecha de fin para el reporte"),
):
    try:
        if not fecha_inicio:
            fecha_fin = date.today()
            fecha_inicio = fecha_fin - timedelta(days=30)
        elif not fecha_fin:
            fecha_fin = date.today()

        if fecha_inicio > fecha_fin:
            fecha_inicio, fecha_fin = fecha_fin, fecha_inicio

        total_protestas = db.query(Protesta).filter(
            Protesta.soft_delete == False,
            Protesta.fecha_evento.between(fecha_inicio, fecha_fin)
        ).count()
        
        total_usuarios = db.query(Usuario).filter(Usuario.soft_delete == False).count()
        total_naturalezas = db.query(Naturaleza).filter(Naturaleza.soft_delete == False).count()
        total_cabecillas = db.query(Cabecilla).filter(Cabecilla.soft_delete == False).count()

        protestas_recientes = (
            db.query(Protesta)
            .filter(
                Protesta.soft_delete == False,
                Protesta.fecha_evento.between(fecha_inicio, fecha_fin)
            )
            .order_by(Protesta.fecha_creacion.desc())
            .limit(5)
            .all()
        )

        protestas_recientes_formatted = []
        for protesta in protestas_recientes:
            try:
                protestas_recientes_formatted.append({
                    "id": str(protesta.id),
                    "nombre": protesta.nombre,
                    "fecha_evento": protesta.fecha_evento.strftime("%Y-%m-%d") if protesta.fecha_evento else None,
                    "fecha_creacion": protesta.fecha_creacion.strftime("%Y-%m-%d") if protesta.fecha_creacion else None,
                })
            except Exception as e:
                logger.error(f"Error al formatear protesta: {protesta.id}, Error: {str(e)}")

        protestas_por_naturaleza = dict(
            db.query(Naturaleza.nombre, func.count(Protesta.id))
            .join(Protesta)
            .filter(
                Protesta.soft_delete == False,
                Protesta.fecha_evento.between(fecha_inicio, fecha_fin)
            )
            .group_by(Naturaleza.nombre)
            .all()
        )

        protestas_por_provincia = dict(
            db.query(Provincia.nombre, func.count(Protesta.id))
            .join(Protesta)
            .filter(
                Protesta.soft_delete == False,
                Protesta.fecha_evento.between(fecha_inicio, fecha_fin)
            )
            .group_by(Provincia.nombre)
            .all()
        )

        protestas_por_dia = dict(
            db.query(func.date(Protesta.fecha_evento), func.count(Protesta.id))
            .filter(
                Protesta.soft_delete == False,
                Protesta.fecha_evento.between(fecha_inicio, fecha_fin)
            )
            .group_by(func.date(Protesta.fecha_evento))
            .order_by(func.date(Protesta.fecha_evento))
            .all()
        )

        top_cabecillas = [
            {"nombre": f"{nombre} {apellido}", "total_protestas": total}
            for nombre, apellido, total in db.query(
                Cabecilla.nombre, Cabecilla.apellido, 
                func.count(ProtestaCabecilla.protesta_id).label('total_protestas')
            )
            .join(ProtestaCabecilla)
            .join(Protesta)
            .filter(
                Protesta.soft_delete == False,
                Protesta.fecha_evento.between(fecha_inicio, fecha_fin)
            )
            .group_by(Cabecilla.id)
            .order_by(func.count(ProtestaCabecilla.protesta_id).desc())
            .limit(10)
            .all()
        ]

        usuarios_activos = [
            {"nombre": f"{nombre} {apellidos}", "protestas_creadas": total}
            for nombre, apellidos, total in db.query(
                Usuario.nombre, Usuario.apellidos, 
                func.count(Protesta.id).label('protestas_creadas')
            )
            .join(Protesta, Protesta.creado_por == Usuario.id)
            .filter(
                Protesta.soft_delete == False,
                Protesta.fecha_evento.between(fecha_inicio, fecha_fin)
            )
            .group_by(Usuario.id)
            .order_by(func.count(Protesta.id).desc())
            .limit(5)
            .all()
        ]

        return {
            "totales": {
                "protestas": total_protestas,
                "usuarios": total_usuarios,
                "naturalezas": total_naturalezas,
                "cabecillas": total_cabecillas
            },
            "protestas_recientes": protestas_recientes_formatted,
            "protestas_por_naturaleza": protestas_por_naturaleza,
            "protestas_por_provincia": protestas_por_provincia,
            "protestas_por_dia": {
                fecha.strftime("%Y-%m-%d") if isinstance(fecha, datetime) else str(fecha): str(count)
                for fecha, count in protestas_por_dia.items()
            },
            "top_cabecillas": top_cabecillas,
            "usuarios_activos": usuarios_activos,
            "fecha_inicio": fecha_inicio.isoformat(),
            "fecha_fin": fecha_fin.isoformat(),
        }

    except Exception as e:
        logger.error(f"Error al obtener resumen principal: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.put("/usuarios/{usuario_id}/rol", response_model=UsuarioSalida)
def cambiar_rol_usuario(
    usuario_id: uuid.UUID,
    nuevo_rol: str = Query(..., pattern="^(admin|usuario)$"),
    usuario_actual: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

        if usuario.rol == "admin" and nuevo_rol == "usuario":
            admin_count = db.query(Usuario).filter(Usuario.rol == "admin", Usuario.soft_delete == False).count()
            if admin_count == 1:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No se puede cambiar el rol del último administrador")

        if usuario.id == usuario_actual.id and nuevo_rol != usuario_actual.rol:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No puedes cambiar tu propio rol")

        if usuario.rol != nuevo_rol:
            usuario.rol = nuevo_rol
            db.commit()
            db.refresh(usuario)
            logger.info(f"{usuario_actual.nombre} {usuario_actual.apellidos} actualizó exitosamente el Rol del usuario: {usuario.email} - Nuevo rol: {nuevo_rol}")

        return usuario
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al cambiar rol de usuario: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Nuevas rutas para cambio de contraseña
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.put("/usuarios/{usuario_id}/cambiar-contrasena")
def cambiar_contrasena_usuario(
    usuario_id: uuid.UUID,
    datos: CambioContrasenaUsuario,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db)
):
    try:
        if usuario_actual.id != usuario_id:
            raise HTTPException(status_code=403, detail="No tienes permiso para cambiar la contraseña de otro usuario")
        
        if not pwd_context.verify(datos.contrasena_actual, usuario_actual.password):
            raise HTTPException(status_code=400, detail="La contraseña actual es incorrecta")
        
        if pwd_context.verify(datos.nueva_contrasena, usuario_actual.password):
            raise HTTPException(status_code=400, detail="La nueva contraseña debe ser diferente de la actual")
        
        try:
            validar_contrasena(datos.nueva_contrasena)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if datos.nueva_contrasena != datos.confirmar_contrasena:
            raise HTTPException(status_code=400, detail="La nueva contraseña y la confirmación no coinciden")
        
        hash_nueva_contrasena = pwd_context.hash(datos.nueva_contrasena)
        usuario_actual.password = hash_nueva_contrasena
        db.commit()
        
        return {"mensaje": "Contraseña actualizada exitosamente"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error interno al cambiar contraseña: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/admin/usuarios/{usuario_id}/restablecer-contrasena")
def restablecer_contrasena_admin(
    usuario_id: uuid.UUID,
    datos: RestablecerContrasenaAdmin,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db)
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        hash_nueva_contrasena = obtener_hash_password(datos.nueva_contrasena)
        usuario.password = hash_nueva_contrasena
        db.commit()
        
        return {"mensaje": f"Contraseña restablecida exitosamente para el usuario {usuario.email}"}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.get("/usuarios", response_model=List[UsuarioSalida])
def listar_usuarios(
    admin_actual: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        usuarios = db.query(Usuario).filter(
            Usuario.soft_delete == False,
            Usuario.email != "admin@test.com"  # Filtrar el usuario admin@test.com
        ).all()
        return usuarios
    except Exception as e:
        logger.error(f"Error al listar usuarios: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")

@app.get("/usuarios/me", response_model=UsuarioSalida)
async def obtener_usuario_actual_ruta(
    usuario: Usuario = Depends(obtener_usuario_actual),
):
    usuario_salida = UsuarioSalida.model_validate(usuario)
    return usuario_salida

def verificar_autenticacion(usuario: Usuario = Depends(obtener_usuario_actual)):
    if not usuario:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")
    return usuario

# Nuevas rutas admin REGISTRO
@app.post("/admin/usuarios", response_model=UsuarioSalida)
async def crear_usuario_admin(
    usuario_data: Tuple[CrearUsuario, Optional[UploadFile]] = Depends(crear_usuario_form),
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    usuario, foto = usuario_data
    logger.info(f"Intento de creación de usuario por admin: {admin.email}")
    logger.debug(f"Datos recibidos: {usuario}")
    try:
        db_usuario = await crear_usuario(usuario, foto, db, is_admin_creation=True)
        logger.info(f"Usuario '{usuario.email}' creado por admin '{admin.email}'")
        return UsuarioSalida.model_validate(db_usuario)
    except HTTPException as he:
        logger.error(f"Error HTTP al crear usuario: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error al crear usuario admin: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")

@app.delete("/admin/usuarios/{usuario_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_usuario_admin(
    usuario_id: uuid.UUID,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        if usuario_id == admin.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No puedes eliminar tu propio usuario")

        db_usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not db_usuario:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

        db_usuario.soft_delete = True
        db.commit()
        logger.info(f"Usuario eliminado exitosamente por admin: {db_usuario.email}")
        return {"detail": "Usuario eliminado exitosamente"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar usuario: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")

@app.post("/registro", response_model=UsuarioSalida)
async def registrar_usuario(
    usuario_data: Tuple[CrearUsuario, Optional[UploadFile]] = Depends(crear_usuario_form),
    db: Session = Depends(obtener_db),
):
    usuario, foto = usuario_data
    return await crear_usuario(usuario, foto, db)

@app.post("/token", response_model=Token)
async def login_para_token_acceso(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(obtener_db)
):
    logger.info(f"Intento de inicio de sesión para: {form_data.username}")
    usuario = db.query(Usuario).filter(Usuario.email == form_data.username, Usuario.soft_delete == False).first()
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
        tipo_token="bearer",
    )

@app.post("/token/renovar", response_model=Token)
async def renovar_token(
    token_actualizacion: str = Body(..., embed=True), db: Session = Depends(obtener_db)
):
    try:
        logger.info("Intento de renovación de token")
        logger.debug(f"Token de actualización recibido: {token_actualizacion}")

        try:
            payload = jwt.decode(
                token_actualizacion, CLAVE_SECRETA, algorithms=[ALGORITMO],
                options={"verify_exp": True},
                leeway=10  
            )
            logger.debug(f"Payload decodificado: {payload}")
        except jwt.ExpiredSignatureError:
            logger.warning("Token de actualización expirado")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token de actualización expirado. Por favor, inicie sesión nuevamente."
            )
        except jwt.JWTError as e:
            logger.error(f"Error al decodificar el token: {str(e)}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

        email: str = payload.get("sub")

        if email is None:
            logger.warning(f"Payload del token inválido: {payload}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token de actualización inválido")

        usuario = db.query(Usuario).filter(Usuario.email == email).first()
        if usuario is None:
            logger.warning(f"Usuario no encontrado para el email: {email}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado")

        nuevo_token_acceso = crear_token_acceso({"sub": email})
        nuevo_token_actualizacion = crear_token_actualizacion({"sub": email})

        logger.info(f"Token renovado exitosamente para: {email}")
        return Token(
            token_acceso=nuevo_token_acceso,
            token_actualizacion=nuevo_token_actualizacion,
            tipo_token="bearer",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error inesperado durante la renovación del token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor. Por favor, intente nuevamente más tarde.",
        )

@app.post("/naturalezas", response_model=NaturalezaSalida)
def crear_naturaleza(
    naturaleza: CrearNaturaleza,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        db_naturaleza = Naturaleza(**naturaleza.model_dump(), creado_por=usuario.id)
        db.add(db_naturaleza)
        db.commit()
        db.refresh(db_naturaleza)

        logger.info(f"Naturaleza creada exitosamente: '{naturaleza.nombre}', por USUARIO: {usuario.nombre} {usuario.apellidos}")

        return NaturalezaSalida.from_orm(db_naturaleza)

    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "naturalezas_nombre_key" in str(e.orig):
                logger.warning(f"Intento de crear naturaleza con nombre duplicado: {naturaleza.nombre}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ya existe una naturaleza con el nombre '{naturaleza.nombre}'")
        logger.error(f"Error de integridad al crear naturaleza: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error al crear la naturaleza. Por favor, intente de nuevo.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear naturaleza: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")

@app.post("/cabecillas", response_model=CabecillaSalida)
async def crear_cabecilla(
    nombre: str = Form(...),
    apellido: str = Form(...),
    cedula: str = Form(...),
    telefono: Optional[str] = Form(None),
    direccion: Optional[str] = Form(None),
    foto: Optional[UploadFile] = File(None),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
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
        db.flush()

        if foto:
            file_extension = os.path.splitext(foto.filename)[1]
            file_name = f"cabecilla_{db_cabecilla.id}{file_extension}"
            relative_path = os.path.join(UPLOAD_DIR, file_name)
            file_path = os.path.join(STATIC_FILES_DIR, relative_path)

            if await save_upload_file(foto, file_path):
                db_cabecilla.foto = relative_path
            else:
                db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error al guardar la imagen")

        db.commit()
        db.refresh(db_cabecilla)

        logger.info(f"Cabecilla '{db_cabecilla.nombre} {db_cabecilla.apellido}' creado exitosamente por usuario: '{usuario.email}'")

        return CabecillaSalida.model_validate(db_cabecilla)

    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "cabecillas_cedula_key" in str(e.orig):
                logger.warning(f"Intento de crear cabecilla con cédula duplicada: {cedula}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ya existe un cabecilla con la cédula '{cedula}'")
        logger.error(f"Error de integridad al crear cabecilla: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error al crear el cabecilla. Por favor, intente de nuevo.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear cabecilla: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")

@app.post("/protestas/completa", response_model=ProtestaSalida)
def crear_protesta_completa(
    protesta: CrearProtestaCompleta,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        if protesta.nueva_naturaleza:
            nueva_naturaleza = Naturaleza(**protesta.nueva_naturaleza.model_dump(), creado_por=usuario.id)
            db.add(nueva_naturaleza)
            db.flush()
            naturaleza_id = nueva_naturaleza.id
        else:
            naturaleza_id = protesta.naturaleza_id

        nuevos_cabecillas_ids = []
        for nuevo_cabecilla in protesta.nuevos_cabecillas:
            db_cabecilla = Cabecilla(**nuevo_cabecilla.model_dump(), creado_por=usuario.id)
            db.add(db_cabecilla)
            db.flush()
            nuevos_cabecillas_ids.append(db_cabecilla.id)

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

        for cabecilla_id in protesta.cabecillas + nuevos_cabecillas_ids:
            cabecilla = db.query(Cabecilla).get(cabecilla_id)
            if cabecilla:
                db_protesta.cabecillas.append(cabecilla)
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Cabecilla con ID {cabecilla_id} no encontrado")

        db.commit()
        db.refresh(db_protesta)

        return ProtestaSalida.model_validate(db_protesta)

    except IntegrityError as e:
        db.rollback()
        logger.error(f"Error de integridad al crear la protesta: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error al crear la protesta. Por favor, intente de nuevo.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear protesta: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")

@app.post("/protestas", response_model=ProtestaSalida)
def crear_protesta(
    protesta: CrearProtesta,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        naturaleza = db.query(Naturaleza).get(protesta.naturaleza_id)
        provincia = db.query(Provincia).get(protesta.provincia_id)
        if not naturaleza or not provincia:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Naturaleza o provincia no válida")

        cabecillas_ids = protesta.cabecillas
        cabecillas = db.query(Cabecilla).filter(Cabecilla.id.in_(cabecillas_ids)).all()
        if len(cabecillas) != len(cabecillas_ids):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uno o más cabecillas no son válidos")

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

        for cabecilla_id in cabecillas_ids:
            cabecilla = db.query(Cabecilla).get(cabecilla_id)
            if cabecilla:
                db_protesta.cabecillas.append(cabecilla)

        db.commit()
        db.refresh(db_protesta)
        logger.info(f"Protesta creada exitosamente: {protesta.nombre}")
        return ProtestaSalida.model_validate(db_protesta)

    except HTTPException as he:
        raise he
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Error de integridad al crear la protesta: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error de integridad en la base de datos. Por favor, intente de nuevo.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear protesta: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error interno del servidor: {str(e)}")

@app.get("/protestas", response_model=PaginatedResponse[ProtestaSalida])
def obtener_protestas(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    fecha_desde: Optional[date] = None,
    fecha_hasta: Optional[date] = None,
    provincia_id: Optional[uuid.UUID] = None,
    naturaleza_id: Optional[uuid.UUID] = None,
    cabecilla_ids: Optional[str] = Query(None, description="Comma-separated list of cabecilla IDs"),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    query = (
        db.query(Protesta)
        .join(Usuario, Protesta.creado_por == Usuario.id)
        .options(
            joinedload(Protesta.naturaleza),
            joinedload(Protesta.provincia),
            joinedload(Protesta.cabecillas),
            contains_eager(Protesta.creador)
        )
        .filter(Protesta.soft_delete == False)
    )

    if fecha_desde:
        query = query.filter(Protesta.fecha_evento >= fecha_desde)

    if fecha_hasta:
        query = query.filter(Protesta.fecha_evento <= fecha_hasta)

    if provincia_id:
        query = query.filter(Protesta.provincia_id == provincia_id)

    if naturaleza_id:
        query = query.filter(Protesta.naturaleza_id == naturaleza_id)

    if cabecilla_ids:
        cabecilla_id_list = [uuid.UUID(id.strip()) for id in cabecilla_ids.split(',')]
        for cabecilla_id in cabecilla_id_list:
            query = query.filter(Protesta.cabecillas.any(Cabecilla.id == cabecilla_id))

    total = query.count()

    protestas = query.order_by(Protesta.fecha_evento.desc()).offset((page - 1) * page_size).limit(page_size).all()

    protestas_salida = [
        ProtestaSalida.model_validate({
            **protesta.__dict__,
            'creador_nombre': f"{protesta.creador.nombre} {protesta.creador.apellidos}",
            'creador_email': protesta.creador.email
        })
        for protesta in protestas
    ]

    return {
        "items": protestas_salida,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }

@app.get("/protestas/{protesta_id}", response_model=ProtestaSalida)
def obtener_protesta(
    protesta_id: str,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        protesta_uuid = uuid.UUID(protesta_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de protesta inválido")

    protesta = (
        db.query(Protesta)
        .join(Usuario, Protesta.creado_por == Usuario.id)
        .options(
            joinedload(Protesta.naturaleza),
            joinedload(Protesta.provincia),
            joinedload(Protesta.cabecillas),
            contains_eager(Protesta.creador)
        )
        .filter(Protesta.id == protesta_uuid, Protesta.soft_delete == False)
        .first()
    )

    if not protesta:
        raise HTTPException(status_code=404, detail="Protesta no encontrada")

    return ProtestaSalida.model_validate({
        **protesta.__dict__,
        'creador_nombre': f"{protesta.creador.nombre} {protesta.creador.apellidos}",
        'creador_email': protesta.creador.email
    })

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

        verificar_propiedad(db_protesta, usuario)

        db_protesta.nombre = protesta.nombre
        db_protesta.resumen = protesta.resumen
        db_protesta.fecha_evento = protesta.fecha_evento

        db_protesta.naturaleza = db.query(Naturaleza).get(protesta.naturaleza_id)
        db_protesta.provincia = db.query(Provincia).get(protesta.provincia_id)

        db_protesta.cabecillas = []
        for cabecilla_id in protesta.cabecillas:
            db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id).first()
            if db_cabecilla:
                db_protesta.cabecillas.append(db_cabecilla)
            else:
                raise HTTPException(status_code=400, detail=f"Cabecilla con ID {cabecilla_id} no encontrado")

        db.commit()
        db.refresh(db_protesta)
        logger.info(f"Protesta actualizada exitosamente: {db_protesta.nombre}")
        return ProtestaSalida.model_validate(db_protesta)

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar protesta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.delete("/protestas/{protesta_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_protesta(
    protesta_id: uuid.UUID,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id).first()
        if not db_protesta:
            raise HTTPException(status_code=404, detail="Protesta no encontrada")

        db_protesta.soft_delete = True
        db.commit()

        logger.info(f"Protesta '{db_protesta.nombre}' eliminada exitosamente por admin: {admin.nombre} {admin.apellidos}")

        return {"detail": "Protesta eliminada exitosamente"}

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar protesta: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/provincias", response_model=List[ProvinciaSalida])
def obtener_provincias(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        provincias = db.query(Provincia).filter(Provincia.soft_delete == False).all()
        logger.info(f"Provincias obtenidas exitosamente. Total: {len(provincias)}")
        return provincias
    except Exception as e:
        logger.error(f"Error al obtener provincias: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/provincias/{provincia_id}", response_model=ProvinciaSalida)
def obtener_provincia(
    provincia_id: uuid.UUID,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        provincia = db.query(Provincia).filter(Provincia.id == provincia_id, Provincia.soft_delete == False).first()
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
            "pages": (total + page_size - 1) // page_size,
        }

        logger.info(f"Naturalezas obtenidas exitosamente. Total: {total}, Página: {page}")
        return result
    except Exception as e:
        logger.error(f"Error al obtener naturalezas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def obtener_naturaleza(
    naturaleza_id: uuid.UUID,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id, Naturaleza.soft_delete == False).first()
        if not naturaleza:
            logger.warning(f"Naturaleza no encontrada: {naturaleza_id}")
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")

        logger.info(f"Naturaleza obtenida exitosamente: {naturaleza.nombre}")
        return NaturalezaSalida.from_orm(naturaleza)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error al obtener naturaleza: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/naturalezas/{naturaleza_id}", response_model=NaturalezaSalida)
def actualizar_naturaleza(
    naturaleza_id: uuid.UUID,
    naturaleza: CrearNaturaleza,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        db_naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id).first()
        if not db_naturaleza:
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")

        verificar_propiedad(db_naturaleza, usuario)

        for key, value in naturaleza.model_dump().items():
            setattr(db_naturaleza, key, value)

        db.commit()
        db.refresh(db_naturaleza)
        logger.info(f"Naturaleza '{db_naturaleza.nombre}' actualizada exitosamente por usuario: '{usuario.email}'")
        return NaturalezaSalida.from_orm(db_naturaleza)

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar naturaleza: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/naturalezas/{naturaleza_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_naturaleza(
    naturaleza_id: uuid.UUID,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        db_naturaleza = db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id).first()
        if not db_naturaleza:
            logger.warning(f"Naturaleza no encontrada: {naturaleza_id}")
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")

        protestas_asociadas = db.query(Protesta).filter(
            Protesta.naturaleza_id == naturaleza_id,
            Protesta.soft_delete == False,
        ).first()

        if protestas_asociadas:
            logger.warning(f"Naturaleza {naturaleza_id} asociada a protestas activas.")
            raise HTTPException(
                status_code=400,
                detail="No se puede eliminar una naturaleza asociada a protestas. Elimine o edite las protestas primero.",
            )

        db_naturaleza.soft_delete = True
        db.commit()

        logger.info(f"Naturaleza {db_naturaleza.nombre} eliminada exitosamente por admin: {admin.nombre} {admin.apellidos}")

        return {"detail": "Naturaleza eliminada exitosamente"}

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar naturaleza: {str(e)}")
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
        cabecillas_salida = [CabecillaSalida.model_validate(c.__dict__) for c in cabecillas]

        result = {
            "items": cabecillas_salida,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
        }

        logger.info(f"Cabecillas obtenidos exitosamente. Total: {total}, Página: {page}")
        return result
    except Exception as e:
        logger.error(f"Error al obtener cabecillas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/cabecillas/all", response_model=List[CabecillaSalida])
def obtener_todos_los_cabecillas(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        cabecillas = db.query(Cabecilla).filter(Cabecilla.soft_delete == False).all()
        cabecillas_salida = [CabecillaSalida.model_validate(c.__dict__) for c in cabecillas]
        return cabecillas_salida
    except Exception as e:
        logger.error(f"Error al obtener todos los cabecillas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
def obtener_cabecilla(
    cabecilla_id: uuid.UUID,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.soft_delete == False).first()
        if not cabecilla:
            logger.warning(f"Cabecilla no encontrado: {cabecilla_id}")
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")
        logger.info(f"Cabecilla obtenido exitosamente: {cabecilla.nombre} {cabecilla.apellido}")
        return CabecillaSalida.model_validate(cabecilla)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error al obtener cabecilla: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/cabecillas/{cabecilla_id}", response_model=CabecillaSalida)
async def actualizar_cabecilla(
    cabecilla_id: uuid.UUID,
    nombre: Optional[str] = Form(None),
    apellido: Optional[str] = Form(None),
    cedula: Optional[str] = Form(None),
    telefono: Optional[str] = Form(None),
    direccion: Optional[str] = Form(None),
    foto: Optional[UploadFile] = File(None),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id, Cabecilla.soft_delete == False).first()
        if not db_cabecilla:
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")

        verificar_propiedad(db_cabecilla, usuario)

        if nombre is not None:
            db_cabecilla.nombre = nombre
        if apellido is not None:
            db_cabecilla.apellido = apellido
        if cedula is not None:
            db_cabecilla.cedula = cedula
        if telefono is not None:
            db_cabecilla.telefono = telefono
        if direccion is not None:
            db_cabecilla.direccion = direccion

        if foto:
            new_foto_url = await actualizar_foto(cabecilla_id, foto, db, "cabecilla", usuario.id)
            db_cabecilla.foto = new_foto_url

        db.commit()
        db.refresh(db_cabecilla)

        logger.info(f"Cabecilla actualizado exitosamente: {db_cabecilla.nombre} {db_cabecilla.apellido}")
        return CabecillaSalida.model_validate(db_cabecilla)
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar cabecilla: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/all-data", response_model=AllData)
def obtener_todos_los_datos(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    fecha_desde: Optional[date] = None,
    fecha_hasta: Optional[date] = None,
    provincia_id: Optional[uuid.UUID] = None,
    naturaleza_id: Optional[uuid.UUID] = None,
    cabecilla_ids: Optional[str] = Query(None, description="Comma-separated list of cabecilla IDs"),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        # Obtener protestas
        protestas_query = (
            db.query(Protesta)
            .join(Usuario, Protesta.creado_por == Usuario.id)
            .options(
                joinedload(Protesta.naturaleza),
                joinedload(Protesta.provincia),
                joinedload(Protesta.cabecillas),
                contains_eager(Protesta.creador)
            )
            .filter(Protesta.soft_delete == False)
        )

        if fecha_desde:
            protestas_query = protestas_query.filter(Protesta.fecha_evento >= fecha_desde)

        if fecha_hasta:
            protestas_query = protestas_query.filter(Protesta.fecha_evento <= fecha_hasta)

        if provincia_id:
            protestas_query = protestas_query.filter(Protesta.provincia_id == provincia_id)

        if naturaleza_id:
            protestas_query = protestas_query.filter(Protesta.naturaleza_id == naturaleza_id)

        if cabecilla_ids:
            cabecilla_id_list = [uuid.UUID(id.strip()) for id in cabecilla_ids.split(',')]
            for cabecilla_id in cabecilla_id_list:
                protestas_query = protestas_query.filter(Protesta.cabecillas.any(Cabecilla.id == cabecilla_id))

        total_protestas = protestas_query.count()
        protestas = protestas_query.order_by(Protesta.fecha_evento.desc()).offset((page - 1) * page_size).limit(page_size).all()
        protestas_salida = [
            ProtestaSalida.model_validate({
                **protesta.__dict__,
                'creador_nombre': f"{protesta.creador.nombre} {protesta.creador.apellidos}",
                'creador_email': protesta.creador.email
            })
            for protesta in protestas
        ]

        # Obtener naturalezas
        naturalezas = db.query(Naturaleza).filter(Naturaleza.soft_delete == False).all()
        naturalezas_salida = [NaturalezaSalida.from_orm(n) for n in naturalezas]

        # Obtener provincias
        provincias = db.query(Provincia).filter(Provincia.soft_delete == False).all()
        provincias_salida = [ProvinciaSalida.model_validate(p) for p in provincias]

        # Obtener cabecillas
        cabecillas = db.query(Cabecilla).filter(Cabecilla.soft_delete == False).all()
        cabecillas_salida = [CabecillaSalida.model_validate(c.__dict__) for c in cabecillas]

        return {
            "protestas": {
                "items": protestas_salida,
                "total": total_protestas,
                "page": page,
                "page_size": page_size,
                "pages": (total_protestas + page_size - 1) // page_size,
            },
            "naturalezas": naturalezas_salida,
            "provincias": provincias_salida,
            "cabecillas": cabecillas_salida,
        }
    except Exception as e:
        logger.error(f"Error al obtener todos los datos: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/cabecillas/{cabecilla_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_cabecilla(
    cabecilla_id: uuid.UUID,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id).first()
        if not db_cabecilla:
            logger.warning(f"Cabecilla no encontrado: {cabecilla_id}")
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")

        protestas_asociadas = db.query(Protesta).join(ProtestaCabecilla).filter(
            ProtestaCabecilla.cabecilla_id == cabecilla_id,
            Protesta.soft_delete == False
            ).first()

        if protestas_asociadas:
            logger.warning(f"Intento de eliminar cabecilla {cabecilla_id} asociado a protestas activas")
            raise HTTPException(
                status_code=400, 
                detail="No se puede eliminar el cabecilla porque está asociado a una o más protestas activas. " 
                       "Elimine o edite las protestas asociadas antes de eliminar el cabecilla."
            )

        db_cabecilla.soft_delete = True
        db.commit()
        logger.info(
            f"Cabecilla {db_cabecilla.nombre} {db_cabecilla.apellido} eliminado exitosamente por admin: {admin.nombre} {admin.apellidos}"
        )
        return {"detail": "Cabecilla eliminado exitosamente"}
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al eliminar cabecilla: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/usuarios/foto", response_model=UsuarioSalida)
async def actualizar_foto_usuario(
    foto: UploadFile = File(...),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    new_foto_url = await actualizar_foto(usuario.id, foto, db, "usuario", usuario.id)
    usuario.foto = new_foto_url
    return UsuarioSalida.model_validate(usuario)

@app.post("/cabecillas/{cabecilla_id}/foto", response_model=CabecillaSalida)
async def actualizar_foto_cabecilla(
    cabecilla_id: uuid.UUID,
    foto: UploadFile = File(...),
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    new_foto_url = await actualizar_foto(
        cabecilla_id, foto, db, "cabecilla", usuario.id
    )
    cabecilla = db.query(Cabecilla).get(cabecilla_id)
    cabecilla.foto = new_foto_url
    return CabecillaSalida.model_validate(cabecilla)

# Configuración de archivos estáticos
STATIC_FILES_DIR = os.getenv("STATIC_FILES_DIR")
UPLOAD_DIR = os.getenv("UPLOAD_DIRECTORY")
MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB'))
MAX_IMAGE_DIMENSION = int(os.getenv('MAX_IMAGE_DIMENSION'))
ALLOWED_IMAGE_TYPES = ('jpeg,jpg,png,gif,webp,bmp').split(',')

if not STATIC_FILES_DIR:
    raise ValueError("La variable de entorno STATIC_FILES_DIR no está configurada.")
if not UPLOAD_DIR:
    raise ValueError("La variable de entorno UPLOAD_DIRECTORY no está configurada.")

os.makedirs(STATIC_FILES_DIR, exist_ok=True)

UPLOAD_DIRECTORY = os.path.join(STATIC_FILES_DIR, UPLOAD_DIR)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

def validate_image(file: UploadFile):
    MAX_IMAGE_SIZE = MAX_IMAGE_SIZE_MB * 1024 * 1024
    
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo es demasiado grande. El tamaño máximo es de {MAX_IMAGE_SIZE_MB} MB.",
        )
    
    contents = file.file.read()
    file.file.seek(0)
    try:
        img = Image.open(io.BytesIO(contents))
        file_type = img.format.lower()
    except IOError:
        raise HTTPException(status_code=400, detail="Archivo de imagen inválido.")
    
    if file_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no permitido. Solo se aceptan {', '.join(ALLOWED_IMAGE_TYPES)}.",
        )
    
    if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Las dimensiones de la imagen son demasiado grandes. El tamaño máximo es de {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} píxeles.",
        )

def get_full_image_url(foto_path: Optional[str]) -> Optional[str]:
    if not foto_path:
        return None
    if foto_path.startswith(('http://', 'https://')):
        return foto_path
    return f"/static/{foto_path}"

async def save_upload_file(upload_file: UploadFile, destination: str) -> bool:
    try:
        contents = await upload_file.read()
        with open(destination, "wb") as buffer:
            buffer.write(contents)
        return True
    except Exception as e:
        logger.error(f"Error al guardar el archivo: {str(e)}")
        return False

def delete_old_file(file_path: str):
    if file_path and os.path.exists(os.path.join(STATIC_FILES_DIR, file_path)):
        os.remove(os.path.join(STATIC_FILES_DIR, file_path))

async def actualizar_foto(
    entity_id: uuid.UUID,
    foto: UploadFile,
    db: Session,
    entity_type: str,
    user_id: uuid.UUID,
) -> str:
    validate_image(foto)
    file_extension = os.path.splitext(foto.filename)[1]
    file_name = f"{entity_type}_{entity_id}{file_extension}"
    relative_path = os.path.join(UPLOAD_DIR, file_name)
    file_path = os.path.join(STATIC_FILES_DIR, relative_path)

    if entity_type == "cabecilla":
        entity = db.query(Cabecilla).filter(Cabecilla.id == entity_id).first()
    elif entity_type == "usuario":
        entity = db.query(Usuario).filter(Usuario.id == entity_id).first()
    else:
        raise ValueError("Tipo de entidad no válido")

    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.capitalize()} no encontrado")

    if entity.foto:
        old_file_path = os.path.join(STATIC_FILES_DIR, entity.foto)
        if os.path.isfile(old_file_path):
            os.remove(old_file_path)

    if await save_upload_file(foto, file_path):
        entity.foto = relative_path
        db.commit()
        db.refresh(entity)
        return relative_path
    else:
        raise HTTPException(status_code=500, detail="Error al guardar la imagen")

# Manejadores de excepciones
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Ha ocurrido un error interno"})

# Configuración del servidor
host = os.getenv("SERVER_HOST")
port = int(os.getenv("SERVER_PORT"))

async def run_server():
    config = uvicorn.Config(app, host=host, port=port, reload=True)
    server = uvicorn.Server(config)
    
    def signal_handler(signum, frame):
        logger.warning("\nDetención solicitada. Cerrando el servidor...")
        asyncio.create_task(server.shutdown())

    signal_module.signal(signal_module.SIGINT, signal_handler)

    try:
        logger.info(f"Iniciando servidor en {host}:{port}...")
        await server.serve()
    except Exception as e:
        logger.error(f"Error al iniciar el servidor: {str(e)}")
    finally:
        logger.info("Servidor detenido.")

if __name__ == "__main__":
    asyncio.run(run_server())
