# Typealiases
Headers = Dict<String, List<String>>
Params = Dict<String, List<String>>
ResponseHandler = (Response) -> Void
MultipartFormDataItems = Array<MultipartFormData>
MultipartFormDataMap = Dict<String, MultipartFormData>
ContentProvider = (offset: Int, len: Int, sink: Datasink) -> bool
ContentProviderWithoutLength = (offset: Int, sink: Datasink) -> bool
ContentProviderResourceReleaser = (bool) -> Void

class MultipartFormData:
    name: String
    content: String
    filename: String 
    content_type: String 

class StatusCode:
    Continue_100 = 100
    ...
    NetworkAuthenticationRequired_511 = 511

class Datasink:
    output_stream: io.BufferedIOBase
    write: (data: Bytes, len: int) -> bool
    is_writable: () -> Bool
    is_done: () -> bool
    done_with_trailed: (Headers) -> bool

# Common
def create_client_socket(scheme, host, port):
    famility, type, proto = ... # from scheme, host, port
    sock = socket(famility, type, proto)
    ... # setup_socket() - set flags and modes
    return sock


## HTTP Client
class HttpClient:
    def __init__(self, scheme_and_port: str):
        self.scheme_and_port = scheme_and_port
        self.scheme, self.host, self.port = ... # <parse_link>
        self.recv_timeout = ... # default
        self.send_timeout = ... # default
        self.keep_connection_alive = ... # default
    
    def create_and_connect_client_socket(self):
        sock = create_client_socket(self.scheme, self.host, self.port)
        ai_addr = ... # from scheme, host, port
        ... # bind(sock, ai_addr)
        ... # set_nonblocking(sock, true)
        ... # connect(sock, ai_addr)
        ... # set_timeouts(sock, self.recv_timeout, self.send_timeout)

        ... # return Socket(sock)
    
    def shutdown_socket(self, socket: Socket):
        ... # shutdown(socket.socket, SHUT_RDWR)
    
    def send(self, req, res):
        is_alive = ... # check_whether_socket_is_alive(self.sock)
        if not is_alive:
            ... # turn_off_socket(self.sock)
            self.sock = self.create_and_connect_client_socket()
        
        ... # setup_default_headers(req)
        self.handle_request(req, res)

    def handle_request(self, req, res):
        req.path = ... # make path from self.host, self.port, req.path
        # TODO