from enum import Enum
from typing import Tuple, AnyStr, List, DefaultDict, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from abc import ABC
import re

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

# Define type aliases
Headers = DefaultDict[AnyStr, List[AnyStr]]
Params = DefaultDict[AnyStr, List[AnyStr]]

class Response:
    ...

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
    
    def is_multiparm_form_data(self) -> bool:
        # TODO: implement
        return False
    
class ClientImpl:
    def __init__(self) -> None:
        pass
class Client:
    scheme_host_port_re = r"((?:([a-z]+):\/\/)?(?:\[([\d:]+)\]|([^:/?#]+))(?::(\d+))?)"
    
    # Add SSL support
    def __init__(self, **kwargs) -> None:
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
            
            client_impl = ClientImpl()

        super().__init__()
    

ResponseHandler = Callable[[Response], bool]

    