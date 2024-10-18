from enum import Enum
from typing import (
    Tuple, AnyStr, 
    List, DefaultDict,
    Callable, TypeAlias, 
    NewType, TypeVar, 
    Optional
)
from collections import defaultdict
from dataclasses import dataclass, field
from abc import ABC
from threading import RLock, Lock, get_ident
import errno

import select
import ctypes
import struct
import fcntl
import re
import socket

class Response:
    ...
class Datasink:
    ...

# Define type aliases
SocketOptions = TypeVar("SocketOptions", Callable[[socket.socket], None])
Headers = TypeVar("Headers", DefaultDict[AnyStr, List[AnyStr]])
Params = TypeVar("Params", DefaultDict[AnyStr, List[AnyStr]])
BindOrConnect = TypeVar("BindOrConnect", Callable[[socket.socket, socket.AddressInfo], bool])
ResponseHandler = TypeVar("ResponseHandler", Callable[[Response], bool])
ContentProvider = TypeVar("ContentProvider", Callable[[int, int, Datasink], bool])
ContentProviderWithoutLength = TypeVar("ContentProviderWithoutLength", Callable[[int, Datasink], bool])
ContentProviderResourceReleaser = TypeVar("ContentProviderResourceReleaser", Callable[[bool], None])


## Utils
def adjust_host_string(host: str) -> str:
    if ":" in host:
        return "[" + host + "]"
    return host

def has_crlf(s: str) -> bool:
    for i in range(len(s)):
        if s[i] == "\r" or s[i] == "\n":
            return True
    return False


## Socket utils
def shutdown_socket(sock: socket.socket) -> int:
    return sock.shutdown(socket.SHUT_RDWR)

def close_socket(sock: socket.socket) -> None:
    return sock.close()

def select_read(sock: socket.socket, sec: int, usec: int) -> int:
    read_list = [sock]
    tv = sec + (usec / 1000)
    fds = select.select(read_list, [], [], tv)
    return len(fds)

def read_socket(sock: socket.socket, length: int, flags: int) -> bytes:
    return sock.recv(length, flags)

def is_socket_alive(sock: socket.socket) -> bool:
    val = select_read(sock, 0, 0)
    if val == 0:
        return True
    elif val < 0 and ctypes.get_errno() == errno.EBADF:
        return False
    return read_socket(sock, 1, socket.MSG_PEEK) > 0

def create_socket(
    host: str,
    ip: str, # optional
    port: int,
    addr_family: int, # From socket module
    socket_flags: int,
    tcp_nodelay: bool,
    socket_options: type[Optional[SocketOptions]],
    bind_or_connect: BindOrConnect
) -> socket.socket:
    address_info = socket.getaddrinfo(
        host, port, 
        addr_family if len(ip) else socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=0,
        flags=socket.AI_NUMERICHOST if len(ip) else socket_flags
    )
    
    for addr in address_info:
        addr_family = addr[0]
        sock_type = addr[1]
        sock_proto = addr[2]
        sock = socket.socket(
            addr_family,
            sock_type,
            sock_proto
        )
        if sock.fileno() == -1:
            raise ValueError("socket fileno -1 on address resolution")
        
        if tcp_nodelay:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        if socket_options:
            socket_options(socket, addr)
        
        if addr_family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        
        if bind_or_connect(socket, addr):
            return sock

        socket.close()
    
    raise ValueError("Failed to create socket")


def create_client_socket(
    host: str,
    ip: str,
    port: int,
    addr_family: int,
    tcp_nodelay: bool,
    socket_options: Callable[[socket.socket], None],
    connection_timeout_sec: int, # Not supported
    connection_timeout_usec: int,
    read_timeout_sec: int,
    read_timeout_usec: int,
    write_timeout_sec: int,
    write_timeout_usec: int,
    intf: str, # Not supported
) -> socket.socket:
    def bind_or_connect(sock: socket.socket, addr) -> bool:
        sock.setblocking(False)
        addrinfo = addr[4]
        sock.connect(addrinfo)

        sock.setblocking(True)
        read_timeout = struct.pack('LL', read_timeout_sec, read_timeout_usec)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, read_timeout)
        
        write_timeout = struct.pack('LL', write_timeout_sec, write_timeout_usec)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, write_timeout)
        
    sock = create_socket(
        host=host, 
        ip=ip, 
        port=port, 
        addr_family=addr_family, 
        socket_flags=0, 
        tcp_nodelay=tcp_nodelay, 
        socket_options=socket_options,
        bind_or_connect=bind_or_connect
    )
    return sock

class StatusCode(Enum):
    Continue_100 = 100
    SwitchingProtocol_101 = 101
    Processing_102 = 102
    EarlyHints_103 = 103

    # Successful responses
    OK_200 = 200
    Created_201 = 201
    Accepted_202 = 202
    NonAuthoritativeInformation_203 = 203
    NoContent_204 = 204
    ResetContent_205 = 205
    PartialContent_206 = 206
    MultiStatus_207 = 207
    AlreadyReported_208 = 208
    IMUsed_226 = 226

    # Redirection messages
    MultipleChoices_300 = 300
    MovedPermanently_301 = 301
    Found_302 = 302
    SeeOther_303 = 303
    NotModified_304 = 304
    UseProxy_305 = 305
    unused_306 = 306
    TemporaryRedirect_307 = 307
    PermanentRedirect_308 = 308

    # Client error responses
    BadRequest_400 = 400
    Unauthorized_401 = 401
    PaymentRequired_402 = 402
    Forbidden_403 = 403
    NotFound_404 = 404
    MethodNotAllowed_405 = 405
    NotAcceptable_406 = 406
    ProxyAuthenticationRequired_407 = 407
    RequestTimeout_408 = 408
    Conflict_409 = 409
    Gone_410 = 410
    LengthRequired_411 = 411
    PreconditionFailed_412 = 412
    PayloadTooLarge_413 = 413
    UriTooLong_414 = 414
    UnsupportedMediaType_415 = 415
    RangeNotSatisfiable_416 = 416
    ExpectationFailed_417 = 417
    ImATeapot_418 = 418
    MisdirectedRequest_421 = 421
    UnprocessableContent_422 = 422
    Locked_423 = 423
    FailedDependency_424 = 424
    TooEarly_425 = 425
    UpgradeRequired_426 = 426
    PreconditionRequired_428 = 428
    TooManyRequests_429 = 429
    RequestHeaderFieldsTooLarge_431 = 431
    UnavailableForLegalReasons_451 = 451

    # Server error responses
    InternalServerError_500 = 500
    NotImplemented_501 = 501
    BadGateway_502 = 502
    ServiceUnavailable_503 = 503
    GatewayTimeout_504 = 504
    HttpVersionNotSupported_505 = 505
    VariantAlsoNegotiates_506 = 506
    InsufficientStorage_507 = 507
    LoopDetected_508 = 508
    NotExtended_510 = 510
    NetworkAuthenticationRequired_511 = 511


@dataclass
class ContentProviderAdapter:
    content_provider: ContentProvider

    def __call__(self, offset: int, _: int, sink: Datasink) -> struct.Any:
        return self.content_provider(offset, sink)


@dataclass
class MultipartFormData:
    name: str
    content: str
    filename: str
    content_type: str

@dataclass
class Request:
    method: str
    path: str
    headers: Headers
    body: str
    remote_addr: str
    remote_port: int
    local_addr: str
    local_port: int

    version: str
    target: str
    params: Params

    content_length: int = field(default=0)
    files: DefaultDict[str, List[MultipartFormData]]


    def has_header(self, key: str) -> bool:
        return key in self.headers

    def get_header_value(self, key: str, id: int = 0) -> str:
        return self.headers[key][id]

    def get_header_value_count(self, key: str) -> int:
        if key not in self.headers:
            return 0
        return len(self.headers[key])
    
    def set_header(self, key: str, value: str) -> None:
        self.headers[key].append(value)
    
    def has_param(self, key: str) -> bool:
        return key in self.params
    
    def get_param_value(self, key: str, id: int = 0) -> str:
        return self.params[key][id]

    def get_param_value_count(self, key: str) -> int:
        if key not in self.params:
            return 0
        return len(self.params[key])
    
    @property
    def is_multiparm_form_data(self) -> bool:
        content_type = self.get_header_value("Content-Type")
        return bool(content_type.find("multipart/form-data", 0))
    
    def has_file(self, key: str) -> bool:
        return key in self.files

    def get_file_value(self, key: str) -> Optional[MultipartFormData]:
        if key in self.files:
            return self.files[key][0]
        return None
    
    def get_files_values(self, key: str) -> List[MultipartFormData]:
        if key in self.files:
            return self.files[key]
        return []


@dataclass
class Response:
    version: str
    status: str = field(default=-1)
    reason: str
    headers: Headers
    body: str
    location: str # Redirect location
    content_length: int = field(default=0)
    content_provider: Optional[ContentProvider] = field(default=None)
    content_provider_resourse_releaser: Optional[ContentProviderResourceReleaser] = field(default=None)
    is_chunked_content_provider: bool = field(default=False);
    content_provider_success_: bool = field(default=False);
    
    def has_header(self, key: str) -> bool:
        return key in self.headers

    def get_header_value(self, key: str, id: int = 0) -> str:
        return self.headers[key][id]

    def get_header_value_count(self, key: str) -> int:
        if key not in self.headers:
            return 0
        return len(self.headers[key])
    
    def set_header(self, key: str, value: str) -> None:
        self.headers[key].append(value)

    def set_redirect(self, url: str, status: StatusCode = StatusCode.Found_302):
        if not has_crlf(url):
            self.set_header("Location", url)
            if 300 <= status.value and status.value <= 400:
                self.status = status
            else:
                self.status = StatusCode.Found_302
    
    def set_content(self, content: str, content_type: str) -> None:
        self.body = content
        self.headers("Content-Type").clear()
        self.set_header("Content-Type", content_type)

    def set_content_provider(
        self, 
        in_length: int,
        content_type: str,
        provider: ContentProvider,
        content_provider_resourse_releaser: ContentProviderResourceReleaser
     ) -> None:
        self.set_header("Content-Type", content_type)
        self.content_length = in_length
        if in_length > 0:
            self.content_provider = provider
        self.content_provider_resourse_releaser = content_provider_resourse_releaser
        self.is_chunked_content_provider = False

    def set_content_provider(
        self, 
        content_type: str,
        provider: ContentProviderWithoutLength,
        content_provider_resourse_releaser: ContentProviderResourceReleaser
     ) -> None:
        self.set_header("Content-Type", content_type)
        self.content_length = 0
        self.content_provider = ContentProviderAdapter(provider)
        self.content_provider_resourse_releaser = content_provider_resourse_releaser
        self.is_chunked_content_provider = False
    
    def set_chunked_content_provider(
        self, 
        content_type: str,
        provider: ContentProviderWithoutLength,
        content_provider_resourse_releaser: ContentProviderResourceReleaser
    ) -> None:
        self.set_header("Content-Type", content_type)
        self.content_length = 0
        self.content_provider = ContentProviderAdapter(provider)
        self.content_provider_resourse_releaser = content_provider_resourse_releaser
        self.is_chunked_content_provider = True


@dataclass
class Socket:
    sock: socket.socket
    # ssl: SSL

    @property
    def is_open(self) -> bool:
        return self.sock.fileno() != -1


class Client:
    scheme_host_port_re = r"((?:([a-z]+):\/\/)?(?:\[([\d:]+)\]|([^:/?#]+))(?::(\d+))?)"
    
    @staticmethod
    def make_client_impl(**kwargs) -> None:
        if (scheme_host_port := kwargs.get("scheme_host_port")):
            smatch = re.search(Client.scheme_host_port_re, scheme_host_port)
            if (schema := smatch.group(1)):
                # TODO: SSL support
                if schema != "http":
                    raise ValueError("Invalid client schema: ", schema)
            # Add check
            is_ssl = False

            host = smatch.group(2) or smatch.group(3)
            port = smatch.group(4)
            port = int(port) if len(port) else \
                443 if (is_ssl) else 80
            
            return ClientImpl(host, port)

class ClientImpl(Client):
    def __init__(
        self,
        host: str, 
        port: str, 
        client_cert_path: str = None,
        client_key_path: str = None
    ) -> None:
        self.host = host
        self.port = port
        self.host_and_port = adjust_host_string(host) + ":" + port
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path

        self.addr_map = dict()
        self.request_mutex = RLock()
        self.socket_mutex = Lock()

        self.socket_should_be_closed_when_request_is_done = False
        self.sock: Socket = None
        self.socket_requests_in_flight = 0
        self.socket_requests_are_from_thread = get_ident()
        self.defeault_headers: Headers = defaultdict(list)
        self.keep_alive = False


    @property
    def is_valid(self):
        return True
    
    def copy_settings(self, rhs: Client) -> None:
        self.client_cert_path = rhs.client_cert_path
        self.client_key_path = rhs.client_key_path
        self.connection_timeout_sec = rhs.connection_timeout_sec
        self.connection_timeout_usec = rhs.connection_timeout_usec
        self.read_timeout_sec = rhs.read_timeout_sec
        self.read_timeout_usec = rhs.read_timeout_usec
        self.write_timeout_sec = rhs.write_timeout_sec
        self.write_timeout_usec = rhs.write_timeout_usec
        self.basic_auth_username = rhs.basic_auth_username
        self.basic_auth_password = rhs.basic_auth_password
        self.bearer_token_auth_token = rhs.bearer_token_auth_token
        #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        self.digest_auth_username = rhs.digest_auth_username
        self.digest_auth_password = rhs.digest_auth_password
        #endif
        self.keep_alive = rhs.keep_alive
        self.follow_location = rhs.follow_location
        self.url_encode = rhs.url_encode
        self.address_family = rhs.address_family
        self.tcp_nodelay = rhs.tcp_nodelay
        self.socket_options = rhs.socket_options
        self.compress = rhs.compress
        self.decompress = rhs.decompress
        self.interface = rhs.interface
        self.proxy_host = rhs.proxy_host
        self.proxy_port = rhs.proxy_port
        self.proxy_basic_auth_username = rhs.proxy_basic_auth_username
        self.proxy_basic_auth_password = rhs.proxy_basic_auth_password
        self.proxy_bearer_token_auth_token = rhs.proxy_bearer_token_auth_token
        #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        self.proxy_digest_auth_username = rhs.proxy_digest_auth_username
        self.proxy_digest_auth_password = rhs.proxy_digest_auth_password
        #endif
        #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        self.ca_cert_file_path = rhs.ca_cert_file_path
        self.ca_cert_dir_path = rhs.ca_cert_dir_path
        self.ca_cert_store = rhs.ca_cert_store
        #endif
        #ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        self.server_certificate_verification = rhs.server_certificate_verification
        #endif
        self.logger = rhs.logger

    def _create_client_socket(self) -> socket.socket:
        if self.proxy_host is not None and len(self.proxy_host) and self.proxy_port is not None and len(self.proxy_port):
            return create_client_socket(
                host=self.proxy_host,
                ip="",
                port=self.proxy_port,
                addr_family=self.address_family,
                tcp_nodelay=self.tcp_nodelay,
                socket_options=self.socket_options,
                connection_timeout_sec=self.connection_timeout_sec,
                connection_timeout_usec=self.connection_timeout_usec,
                read_timeout_sec=self.read_timeout_sec,
                read_timeout_usec=self.read_timeout_usec,
                write_timeout_sec=self.write_timeout_sec,
                write_timeout_usec=self.write_timeout_usec,
                intf="" # not supported
            )
        
        ip = self.addr_map.get(self.host) # Check custom ip for host
        return create_client_socket(
            host=self.host,
            ip=ip,
            port=self.port,
            addr_family=self.address_family,
            tcp_nodelay=self.tcp_nodelay,
            socket_options=self.socket_options,
            connection_timeout_sec=self.connection_timeout_sec,
            connection_timeout_usec=self.connection_timeout_usec,
            read_timeout_sec=self.read_timeout_sec,
            read_timeout_usec=self.read_timeout_usec,
            write_timeout_sec=self.write_timeout_sec,
            write_timeout_usec=self.write_timeout_usec,
            intf="" # not supported
        )
    
    def create_and_connect_socket(self, sock: Socket) -> bool:
        s = self._create_client_socket()
        if s.fileno() == -1: 
            return False
        self.sock.sock = s
        return True

    def set_logger(self, logger: Callable[[Request, Response], None]) -> None:
        self.logger = logger

    def shutdown_socket(self, sock: Socket):
        if self.sock.sock.fileno() == -1:
            return
        shutdown_socket(sock.sock)

    def close_socket(self, sock: Socket) -> None:
        assert(self.socket_requests_in_flight_ == 0) # TODO: Add check
        if sock.sock.fileno() == -1:
            return
        sock.sock = socket.socket()

    def send(self, req: Request, resp: Response) -> bool:
        with self.request_mutex:
            return self._send(req=req, resp=resp)
    
    def _send(self, req: Request, resp: Response) -> bool:
        with self.socket_mutex:
            self.socket_should_be_closed_when_request_is_done = False
            is_alive = False
            if self.sock.is_open:
                is_alive = is_socket_alive(self.sock.sock)
                if not is_alive:
                    shutdown_gracefully = False
                    self.shutdown_ssl(self.sock, shutdown_gracefully)
                    self.shutdown_socket(self.sock)
                    self.close_socket(self.sock)

            if not is_alive:
                if not self.create_and_connect_socket(self.sock):
                    return False
            
            if self.socket_requests_in_flight > 1:
                ...
                # TODO: Add assert
            self.socket_requests_in_flight += 1
            self.socket_requests_are_from_thread = get_ident()

        for (header_key, header_value) in self.default_headers.items():
            if header_key not in req.headers:
                req.headers[header_value].append(header_value)

        ret = False
        close_connection = not self.keep_alive
        