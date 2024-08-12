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
from typing import Generic, List, Optional, Tuple, TypeVar, Dict
from contextlib import asynccontextmanager

# Librerías de terceros
from fastapi import (Body, FastAPI, Depends, File, Form, HTTPException, Query, Request, UploadFile, status)
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr, Field, ValidationError, field_validator, validator
import bcrypt
import uvicorn
from colorama import init, Fore, Style
from dotenv import load_dotenv

# SQLAlchemy
from sqlalchemy import (DateTime, create_engine, Column, String, Boolean, Date, ForeignKey, func, and_)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session, joinedload
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

    naturaleza = relationship("Naturaleza")
    provincia = relationship("Provincia")
    cabecillas = relationship("Cabecilla", secondary="api.protestas_cabecillas")

class ProtestaCabecilla(Base):
    __tablename__ = "protestas_cabecillas"
    __table_args__ = {"schema": "api"}
    protesta_id = Column(GUID(), ForeignKey("api.protestas.id"), primary_key=True)
    cabecilla_id = Column(GUID(), ForeignKey("api.cabecillas.id"), primary_key=True)

# Modelos Pydantic

# update down
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

    class Config:
        from_attributes = True

    @validator("foto", pre=True)
    def get_full_foto_url(cls, v):
        return get_full_image_url(v) if v else None

class CrearUsuario(UsuarioBase):
    password: str
    repetir_password: str

    @field_validator("repetir_password")
    def passwords_match(cls, v, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Las contraseñas no coinciden")
        return v

# update up
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

    @validator("foto", pre=True)
    def get_full_foto_url(cls, v):
        return get_full_image_url(v) if v else None

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

# Creacion y actualizacion de token
def crear_token_acceso(datos: dict) -> str:
    a_codificar = datos.copy()
    expira = datetime.now(timezone.utc) + timedelta(
        minutes=MINUTOS_INACTIVIDAD_PERMITIDOS
    )  # 15 minutos de expiración
    a_codificar.update(
        {
            "exp": expira.timestamp(),
            "ultima_actividad": datetime.now(timezone.utc).isoformat(),
        }
    )
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
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido"
            )

        ultima_actividad = datetime.fromisoformat(ultima_actividad_str)
        tiempo_inactivo = datetime.now(timezone.utc) - ultima_actividad

        if tiempo_inactivo > timedelta(minutes=MINUTOS_INACTIVIDAD_PERMITIDOS):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Sesión expirada por inactividad",
            )

        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado"
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido"
        )

async def obtener_usuario_actual(
    token: str = Depends(oauth2_scheme), db: Session = Depends(obtener_db)
):
    email = await verificar_token_y_actividad(token)
    usuario = db.query(Usuario).filter(Usuario.email == email).first()
    if usuario is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado"
        )
    return usuario

# Funciones de Control de Acceso
def es_admin(usuario: Usuario) -> bool:
    return usuario.rol == "admin"

def verificar_admin(usuario: Usuario = Depends(obtener_usuario_actual)):
    if not es_admin(usuario):
        raise HTTPException(
            status_code=403, detail="Se requieren permisos de administrador"
        )
    return usuario

# Funciones de Paginación
def paginar(
    query, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100)
):
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
async def crear_usuario(
    usuario: CrearUsuario,
    foto: Optional[UploadFile],
    db: Session,
    is_admin_creation: bool = False,
):
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
        print(
            f"Usuario {'creado por admin' if is_admin_creation else 'registrado'} exitosamente: {usuario.email}"
        )
        return db_usuario
    except IntegrityError as e:
        db.rollback()
        if "usuarios_email_key" in str(e.orig):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe un usuario con el email '{usuario.email}'",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al crear el usuario. Por favor, intente de nuevo.",
        )
    except Exception as e:
        db.rollback()
        print(
            f"Error al {'crear' if is_admin_creation else 'registrar'} usuario: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

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

@app.get("/check-user")
def check_user_exists(
    email: str = Query(..., description="Email del usuario a verificar"),
    db: Session = Depends(obtener_db)
):
    usuario = db.query(Usuario).filter(Usuario.email == email, Usuario.soft_delete == False).first()
    return {"exists": usuario is not None}

@app.get("/pagina-principal", response_model=Dict)
def obtener_resumen_principal(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        # Fecha actual y hace 30 días
        hoy = datetime.now().date()
        hace_30_dias = hoy - timedelta(days=30)

        # Total de entidades
        total_protestas = db.query(Protesta).filter(Protesta.soft_delete == False).count()
        total_usuarios = db.query(Usuario).filter(Usuario.soft_delete == False).count()
        total_naturalezas = db.query(Naturaleza).filter(Naturaleza.soft_delete == False).count()
        total_cabecillas = db.query(Cabecilla).filter(Cabecilla.soft_delete == False).count()

        # Protestas recientes
        protestas_recientes = (
            db.query(Protesta)
            .filter(Protesta.soft_delete == False)
            .order_by(Protesta.fecha_creacion.desc())
            .limit(5)
            .all()
        )

        # Protestas por naturaleza
        protestas_por_naturaleza = dict(
            db.query(Naturaleza.nombre, func.count(Protesta.id))
            .join(Protesta)
            .filter(Protesta.soft_delete == False)
            .group_by(Naturaleza.nombre)
            .all()
        )

        # Protestas por provincia
        protestas_por_provincia = dict(
            db.query(Provincia.nombre, func.count(Protesta.id))
            .join(Protesta)
            .filter(Protesta.soft_delete == False)
            .group_by(Provincia.nombre)
            .all()
        )

        # Protestas en los últimos 30 días
        protestas_ultimos_30_dias = {
            fecha.isoformat(): count
            for fecha, count in db.query(func.date(Protesta.fecha_evento), func.count(Protesta.id))
            .filter(and_(Protesta.soft_delete == False, Protesta.fecha_evento >= hace_30_dias))
            .group_by(func.date(Protesta.fecha_evento))
            .order_by(func.date(Protesta.fecha_evento))
            .all()
        }

        # Top 5 cabecillas más activos
        top_cabecillas = [
            {"nombre": f"{nombre} {apellido}", "total_protestas": total}
            for nombre, apellido, total in db.query(
                Cabecilla.nombre, Cabecilla.apellido, 
                func.count(ProtestaCabecilla.protesta_id).label('total_protestas')
            )
            .join(ProtestaCabecilla)
            .join(Protesta)
            .filter(Protesta.soft_delete == False)
            .group_by(Cabecilla.id)
            .order_by(func.count(ProtestaCabecilla.protesta_id).desc())
            .limit(5)
            .all()
        ]

        # Usuarios más activos (creadores de protestas)
        usuarios_activos = [
            {"nombre": f"{nombre} {apellidos}", "protestas_creadas": total}
            for nombre, apellidos, total in db.query(
                Usuario.nombre, Usuario.apellidos, 
                func.count(Protesta.id).label('protestas_creadas')
            )
            .join(Protesta, Protesta.creado_por == Usuario.id)
            .filter(Protesta.soft_delete == False)
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
            "protestas_recientes": [
                {
                    "id": str(protesta.id),
                    "nombre": protesta.nombre,
                    "fecha_evento": protesta.fecha_evento.isoformat(),
                    "fecha_creacion": protesta.fecha_creacion.isoformat(),
                }
                for protesta in protestas_recientes
            ],
            "protestas_por_naturaleza": protestas_por_naturaleza,
            "protestas_por_provincia": protestas_por_provincia,
            "protestas_ultimos_30_dias": protestas_ultimos_30_dias,
            "top_cabecillas": top_cabecillas,
            "usuarios_activos": usuarios_activos
        }

    except Exception as e:
        logger.error(f"Error al obtener resumen principal: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/usuarios/{usuario_id}/rol", response_model=UsuarioSalida)
def cambiar_rol_usuario(
    usuario_id: uuid.UUID,
    nuevo_rol: str = Query(..., regex="^(admin|usuario)$"),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado"
            )

        if usuario_actual.rol != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Solo los administradores pueden cambiar roles",
            )

        if usuario.rol == "admin" and nuevo_rol == "usuario":
            admin_count = (
                db.query(Usuario)
                .filter(Usuario.rol == "admin", Usuario.soft_delete == False)
                .count()
            )
            if admin_count == 1:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No se puede cambiar el rol del último administrador",
                )

        if usuario.id == usuario_actual.id and nuevo_rol != usuario_actual.rol:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No puedes cambiar tu propio rol",
            )

        if usuario.rol != nuevo_rol:
            usuario.rol = nuevo_rol
            db.commit()
            db.refresh(usuario)
            logger.info(
                f"{usuario_actual.nombre} {usuario_actual.apellidos} actualizo exitosamente el Rol del usuario: {usuario.email} - Nuevo rol: {nuevo_rol}"
            )

        return usuario
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        print(f"Error al cambiar rol de usuario: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

# from fastapi.responses import FileResponse

# @app.get("/debug-static/{file_path:path}")
# async def debug_static_files(file_path: str):
#     full_path = os.path.join(STATIC_FILES_DIR, file_path)
#     if os.path.exists(full_path):
#         return FileResponse(full_path)
#     return {"error": "File not found"}

@app.get("/usuarios/me", response_model=UsuarioSalida)
async def obtener_usuario_actual_ruta(
    usuario: Usuario = Depends(obtener_usuario_actual),
):
    usuario_salida = UsuarioSalida.model_validate(usuario)
    # print(f"Ruta de la foto del usuario: {usuario_salida.foto}")
    return usuario_salida

def verificar_autenticacion(usuario: Usuario = Depends(obtener_usuario_actual)):
    if not usuario:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado"
        )
    return usuario

# nuevas rutas admin REGISTRO
@app.post("/admin/usuarios", response_model=UsuarioSalida)
async def crear_usuario_admin(
    usuario_data: Tuple[CrearUsuario, Optional[UploadFile]] = Depends(
        crear_usuario_form
    ),
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

@app.delete("/admin/usuarios/{usuario_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_usuario_admin(
    usuario_id: uuid.UUID,
    admin: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        if usuario_id == admin.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No puedes eliminar tu propio usuario",
            )

        db_usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not db_usuario:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado"
            )

        db_usuario.soft_delete = True
        db.commit()
        logger.info(f"Usuario eliminado exitosamente por admin: {db_usuario.email}")
        return {"detail": "Usuario eliminado exitosamente"}
    except Exception as e:
        db.rollback()
        logger.info(f"Error al eliminar usuario: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

@app.post("/registro", response_model=UsuarioSalida)
async def registrar_usuario(
    usuario_data: Tuple[CrearUsuario, Optional[UploadFile]] = Depends(
        crear_usuario_form
    ),
    db: Session = Depends(obtener_db),
):
    usuario, foto = usuario_data
    return await crear_usuario(usuario, foto, db)

@app.post("/token", response_model=Token)
async def login_para_token_acceso(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(obtener_db)
):
    logger.info(f"Intento de inicio de sesión para: {form_data.username}")
    usuario = (
        db.query(Usuario)
        .filter(Usuario.email == form_data.username, Usuario.soft_delete == False)
        .first()
    )
    if not usuario or not verificar_password(form_data.password, usuario.password):
        logger.warning(
            f"Intento de inicio de sesión fallido para: {form_data.username}"
        )
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
                token_actualizacion, CLAVE_SECRETA, algorithms=[ALGORITMO]
            )
            logger.debug(f"Payload decodificado: {payload}")
        except jwt.JWTError as e:
            logger.error(f"Error al decodificar el token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido"
            )

        email: str = payload.get("sub")
        exp: float = payload.get("exp")

        if email is None or exp is None:
            logger.warning(f"Payload del token inválido: {payload}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token de actualización inválido",
            )

        # Verificar si el token ha expirado
        if datetime.now(timezone.utc).timestamp() > exp:
            logger.warning(f"Token de actualización expirado para: {email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token de actualización expirado",
            )

        usuario = db.query(Usuario).filter(Usuario.email == email).first()
        if usuario is None:
            logger.warning(f"Usuario no encontrado para el email: {email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado"
            )

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
            tipo_token="bearer",
        )
    except Exception as e:
        logger.exception(f"Error inesperado durante la renovación del token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

@app.post("/naturalezas", response_model=NaturalezaSalida)
def crear_naturaleza(
    naturaleza: CrearNaturaleza,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        # Crear la nueva instancia de Naturaleza
        db_naturaleza = Naturaleza(**naturaleza.model_dump(), creado_por=usuario.id)
        db.add(db_naturaleza)
        db.commit()
        db.refresh(db_naturaleza)

        # Log exitoso
        logger.info(
            f"Naturaleza creada exitosamente: '{naturaleza.nombre}', por USUARIO: {usuario.nombre} {usuario.apellidos}"
        )

        return NaturalezaSalida.from_orm(db_naturaleza)

    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "naturalezas_nombre_key" in str(e.orig):
                logger.warning(
                    f"Intento de crear naturaleza con nombre duplicado: {naturaleza.nombre}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ya existe una naturaleza con el nombre '{naturaleza.nombre}'",
                )
        logger.error(f"Error de integridad al crear naturaleza: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al crear la naturaleza. Por favor, intente de nuevo.",
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear naturaleza: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

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

            if await save_upload_file(foto, file_path):
                db_cabecilla.foto = relative_path
            else:
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Error al guardar la imagen",
                )

        db.commit()
        db.refresh(db_cabecilla)

        logger.info(
            f"Cabecilla '{db_cabecilla.nombre} {db_cabecilla.apellido}' creado exitosamente por usuario: '{usuario.email}'"
        )

        # Devolver la respuesta en el formato correcto
        return CabecillaSalida.model_validate(db_cabecilla)

    except IntegrityError as e:
        db.rollback()
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            if "cabecillas_cedula_key" in str(e.orig):
                logger.warning(
                    f"Intento de crear cabecilla con cédula duplicada: {cedula}"
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ya existe un cabecilla con la cédula '{cedula}'",
                )
        logger.error(f"Error de integridad al crear cabecilla: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al crear el cabecilla. Por favor, intente de nuevo.",
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear cabecilla: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

@app.post("/protestas/completa", response_model=ProtestaSalida)
def crear_protesta_completa(
    protesta: CrearProtestaCompleta,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
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
            cabecilla = db.query(Cabecilla).get(cabecilla_id)
            if cabecilla:
                db_protesta.cabecillas.append(cabecilla)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Cabecilla con ID {cabecilla_id} no encontrado",
                )

        db.commit()
        db.refresh(db_protesta)

        return ProtestaSalida.model_validate(db_protesta)

    except IntegrityError as e:
        db.rollback()
        logger.error(f"Error de integridad al crear la protesta: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al crear la protesta. Por favor, intente de nuevo.",
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear protesta: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )

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
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Naturaleza o provincia no válida",
            )

        # Validar que los cabecillas existen
        cabecillas_ids = protesta.cabecillas
        cabecillas = db.query(Cabecilla).filter(Cabecilla.id.in_(cabecillas_ids)).all()
        if len(cabecillas) != len(cabecillas_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uno o más cabecillas no son válidos",
            )

        # Crear la protesta
        db_protesta = Protesta(
            nombre=protesta.nombre,
            naturaleza_id=protesta.naturaleza_id,
            provincia_id=protesta.provincia_id,
            resumen=protesta.resumen,
            fecha_evento=protesta.fecha_evento,
            creado_por=usuario.id,
        )
        db.add(db_protesta)
        db.flush()  # Para obtener el ID de la protesta

        # Asociar cabecillas a la protesta
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error de integridad en la base de datos. Por favor, intente de nuevo.",
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error al crear protesta: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}",
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
    # Convertir el string a UUID
    try:
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

        return ProtestaSalida.model_validate(protesta)

    except Exception as e:
        logger.error(f"Error al obtener protesta: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.put("/protestas/{protesta_id}", response_model=ProtestaSalida)
def actualizar_protesta(
    protesta_id: uuid.UUID,
    protesta: CrearProtesta,
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        # Buscar la protesta existente
        db_protesta = db.query(Protesta).filter(Protesta.id == protesta_id).first()
        if not db_protesta:
            raise HTTPException(status_code=404, detail="Protesta no encontrada")

        if db_protesta.creado_por != usuario.id:
            raise HTTPException(
                status_code=403, detail="No tienes permiso para editar esta protesta"
            )

        # Actualizar campos simples
        db_protesta.nombre = protesta.nombre
        db_protesta.resumen = protesta.resumen
        db_protesta.fecha_evento = protesta.fecha_evento

        db_protesta.naturaleza = db.query(Naturaleza).get(protesta.naturaleza_id)
        db_protesta.provincia = db.query(Provincia).get(protesta.provincia_id)

        db_protesta.cabecillas = []
        for cabecilla_id in protesta.cabecillas:
            db_cabecilla = (
                db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id).first()
            )
            if db_cabecilla:
                db_protesta.cabecillas.append(db_cabecilla)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cabecilla con ID {cabecilla_id} no encontrado",
                )

        db.commit()
        db.refresh(db_protesta)
        print(
            Fore.GREEN
            + f"Protesta actualizada exitosamente: {db_protesta.nombre}"
            + Style.RESET_ALL
        )
        return ProtestaSalida.model_validate(db_protesta)

    except HTTPException as he:
        # Relanzar las excepciones HTTP sin manejar
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar protesta: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error interno del servidor: {str(e)}"
        )

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

        logger.info(
            f"Protesta '{db_protesta.nombre}' eliminada exitosamente por admin: {admin_actual.nombre} {admin_actual.apellidos}"
        )

        return {"detail": "Protesta eliminada exitosamente"}

    except HTTPException as he:
        # Relanzar excepciones HTTP
        raise he
    except Exception as e:
        # Manejar otros errores
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
            "pages": (total + page_size - 1) // page_size,
        }

        logger.info(
            f"Naturalezas obtenidas exitosamente. Total: {total}, Página: {page}"
        )
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
        naturaleza = (
            db.query(Naturaleza)
            .filter(Naturaleza.id == naturaleza_id, Naturaleza.soft_delete == False)
            .first()
        )
        if not naturaleza:
            logger.warning(f"Naturaleza no encontrada: {naturaleza_id}")
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")

        logger.info(f"Naturaleza obtenida exitosamente: {naturaleza.nombre}")
        return NaturalezaSalida.from_orm(naturaleza)
    except HTTPException as he:
        # Relanzar excepciones HTTP
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
        # Verificar si el usuario es admin
        if not es_admin(usuario):
            # Verificar si la naturaleza está asociada a una protesta
            asociada_a_protesta = (
                db.query(Protesta)
                .filter(Protesta.naturaleza_id == naturaleza_id)
                .count()
                > 0
            )

            if asociada_a_protesta:
                logger.warning(
                    f"El usuario {usuario.id} no tiene permiso para editar la naturaleza {naturaleza_id} porque está asociada a una protesta."
                )
                raise HTTPException(
                    status_code=403,
                    detail="No tienes permiso para editar esta naturaleza. Está asociada a una protesta.",
                )

        db_naturaleza = (
            db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id).first()
        )
        if not db_naturaleza:
            logger.warning(f"Naturaleza no encontrada: {naturaleza_id}")
            raise HTTPException(
                status_code=404,
                detail="Naturaleza no encontrada",
            )

        # Actualizar campos
        for key, value in naturaleza.model_dump().items():
            setattr(db_naturaleza, key, value)

        db.commit()
        db.refresh(db_naturaleza)
        logger.info(
            f"Naturaleza '{db_naturaleza.nombre}' actualizada exitosamente por usuario: '{usuario.email}'"
        )
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
    admin_actual: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        # Verificar si la naturaleza existe
        db_naturaleza = (
            db.query(Naturaleza).filter(Naturaleza.id == naturaleza_id).first()
        )
        if not db_naturaleza:
            logger.warning(f"Naturaleza no encontrada: {naturaleza_id}")
            raise HTTPException(status_code=404, detail="Naturaleza no encontrada")

        # Verificar si la naturaleza está asociada a alguna protesta no eliminada
        protestas_asociadas = (
            db.query(Protesta)
            .filter(
                Protesta.naturaleza_id == naturaleza_id,
                Protesta.soft_delete
                == False,  
            )
            .first()
        )

        if protestas_asociadas:
            logger.warning(f"Naturaleza {naturaleza_id} asociada a protestas activas.")
            raise HTTPException(
                status_code=400,
                detail="No se puede eliminar una naturaleza asociada a protestas. Elimine o edite las protestas primero.",
            )

        # Realizar soft delete
        db_naturaleza.soft_delete = True
        db.commit()

        logger.info(
            f"Naturaleza {db_naturaleza.nombre} eliminada exitosamente por admin: {admin_actual.nombre} {admin_actual.apellidos}"
        )

        return {"detail": "Naturaleza eliminada exitosamente"}

    except HTTPException as he:
        # Relanzar excepciones HTTP
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
        # Construir la consulta con filtros
        query = db.query(Cabecilla).filter(Cabecilla.soft_delete == False)

        if nombre:
            query = query.filter(Cabecilla.nombre.ilike(f"%{nombre}%"))
        if apellido:
            query = query.filter(Cabecilla.apellido.ilike(f"%{apellido}%"))
        if cedula:
            query = query.filter(Cabecilla.cedula.ilike(f"%{cedula}%"))

        # Contar el total de registros
        total = query.count()

        # Obtener los resultados paginados
        cabecillas = query.offset((page - 1) * page_size).limit(page_size).all()
        cabecillas_salida = [
            CabecillaSalida.model_validate(c.__dict__) for c in cabecillas
        ]

        result = {
            "items": cabecillas_salida,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
        }

        logger.info(
            f"Cabecillas obtenidos exitosamente. Total: {total}, Página: {page}"
        )
        return result
    except Exception as e:
        logger.error(f"Error al obtener cabecillas: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# NUEVO ENDPOINT
@app.get("/cabecillas/all", response_model=List[CabecillaSalida])
def obtener_todos_los_cabecillas(
    usuario: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_db),
):
    try:
        cabecillas = db.query(Cabecilla).filter(Cabecilla.soft_delete == False).all()
        cabecillas_salida = [
            CabecillaSalida.model_validate(c.__dict__) for c in cabecillas
        ]
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
        cabecilla = (
            db.query(Cabecilla)
            .filter(Cabecilla.id == cabecilla_id, Cabecilla.soft_delete == False)
            .first()
        )
        if not cabecilla:
            logger.warning(f"Cabecilla no encontrado: {cabecilla_id}")
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")
        logger.info(
            f"Cabecilla obtenido exitosamente: {cabecilla.nombre} {cabecilla.apellido}"
        )
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
        db_cabecilla = (
            db.query(Cabecilla)
            .filter(Cabecilla.id == cabecilla_id, Cabecilla.soft_delete == False)
            .first()
        )
        if not db_cabecilla:
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")

        if db_cabecilla.creado_por != usuario.id and usuario.rol != "admin":
            raise HTTPException(
                status_code=403, detail="No tienes permiso para editar este cabecilla"
            )

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
            new_foto_url = await actualizar_foto(
                cabecilla_id, foto, db, "cabecilla", usuario.id
            )
            db_cabecilla.foto = new_foto_url

        db.commit()
        db.refresh(db_cabecilla)

        logger.info(
            f"Cabecilla actualizado exitosamente: {db_cabecilla.nombre} {db_cabecilla.apellido}"
        )
        return CabecillaSalida.model_validate(db_cabecilla)
    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        logger.error(f"Error al actualizar cabecilla: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.delete("/cabecillas/{cabecilla_id}", status_code=status.HTTP_204_NO_CONTENT)
def eliminar_cabecilla(
    cabecilla_id: uuid.UUID,
    usuario: Usuario = Depends(verificar_admin),
    db: Session = Depends(obtener_db),
):
    try:
        db_cabecilla = db.query(Cabecilla).filter(Cabecilla.id == cabecilla_id).first()
        if not db_cabecilla:
            logger.warning(f"Cabecilla no encontrado: {cabecilla_id}")
            raise HTTPException(status_code=404, detail="Cabecilla no encontrado")

        # Eliminar asociaciones con protestas
        db.query(ProtestaCabecilla).filter(
            ProtestaCabecilla.cabecilla_id == cabecilla_id
        ).delete(synchronize_session=False)

        db_cabecilla.soft_delete = True
        db.commit()
        logger.info(
            f"Cabecilla {db_cabecilla.nombre} {db_cabecilla.apellido} eliminado exitosamente por admin: {usuario.nombre} {usuario.apellidos}"
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

# Fin de rutas

# Configurar la ruta base para los archivos estáticos
STATIC_FILES_DIR = "static"
UPLOAD_DIR = os.getenv("UPLOAD_DIRECTORY")
UPLOAD_DIRECTORY = os.path.join(STATIC_FILES_DIR, UPLOAD_DIR)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE"))
ALLOWED_IMAGE_TYPES = os.getenv("ALLOWED_IMAGE_TYPES").split(",")

app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")

# Asegurarse de que el directorio de uploads exista
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Funciones auxiliares
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

def get_full_image_url(foto_path: str) -> str:
    if foto_path:
        return f"/static/{foto_path}"
    return None

async def save_upload_file(upload_file: UploadFile, destination: str) -> bool:
    try:
        contents = await upload_file.read()
        with open(destination, "wb") as buffer:
            buffer.write(contents)
        return True
    except Exception as e:
        print(f"Error al guardar el archivo: {str(e)}")
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
        raise HTTPException(
            status_code=404, detail=f"{entity_type.capitalize()} no encontrado"
        )

    # Eliminar la foto anterior si existe
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
    host=os.getenv("SERVER_HOST", "127.0.0.1"),
    port=int(os.getenv("SERVER_PORT", 8000)),
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
