"""gRPC transport for Dictator speech services."""

from .config import ServerConfig
from .server import build_server, serve

__all__ = ["ServerConfig", "build_server", "serve"]
