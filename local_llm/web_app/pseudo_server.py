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

## HTTP Client
class HTTPClient:
    def __init__(self, **kwargs) -> None:
        host_and_port = adjust_ipv6(kwargs["host"]) + ":" + kwargs["port"]
        client_cert_path = client_cert_path

    def copy_settings(self, rhs):
        ...
    
    def create_socket():
        if has_proxy_host:
            create_socket(proxy_host, proxy_port, socket_args...)

        ip = check_ip_overwrites(host)
        return create_socket(proxy_host, proxy_port, ip, socket_args...)
    
    def create_and_connect_socket():
        return Socket(create_socket())
    
    def read_response_line(Stream strm, req, response):
        line = read_line(strm)
        regex = r"(HTTP/1\\.[01]) (\\d{3})(?: (.*?))?\r\n" # Looking for http header
        smatch = line.match(regex)

        if smatch is None:
            return req.method == "Connect"
        
        res.version = smatch.group(1)
        res.status = smatch.group(2)
        res.reason = smatch.group(3)

        while (res.status = StatusCode::Continue_100): # Response buffered already or not
            if !read_line(strm): return False # CRLF
            next_response_line = read_line(strm)
            if next_response_line is None: return False

            smatch = line.match(next_response_line)
            if smatch is None:
                return False
            res.version = smatch.group(1)
            res.status = smatch.group(2)
            res.reason = smatch.group(3)

        return True
    
    def send(req, resp):
        with request_lock:
            with socket_mutex:
                self.socket_should_be_closed_when_request_is_done_ = False
                is_alive = False
                if sock.is_open():
                    is_alive = self.sock.alive()
                    if !is_socket_alive:
                        shutdown_ssl(socket)
                        shutdown_socket(socket)
                        close_socket(socket)

                if !is_alive:
                    if !create_and_connect_socket(sock):
                        return False
                
                socket_requests_in_flight++
                socket_requests_are_from_thread_ = thread.id()
            
            req.set_default_headers_if_not_set()
            should_close_connection = !self.keep_alive

            ret = 0
            defer {
                with socket_mutex:
                    socket_requests_in_flight--;
                    if should_close_connection || ret:
                        shutdown_ssl(socket)
                        shutdown_socket(socket)
                        close_socket(socket)
            }

            handle_request(req, resp, should_close_connection)

    def write_content_with_provider(strm, req):
        if req.is_chunked_content_provider:


    def write_request(strm, req, close_connection):
        if close_connection:
            req.set_header_if_not("Connection", "close")
        
        req.set_header_if_not("Host", host_ or host_and_port)
        req.set_header_if_not("Accept", "*/*")
        
        if req.body.isEmpty():
            if req.content_provider:
                if !req.content_provider.chunked:
                    req.set_header_if_not("Content-Length", req.content_length)
            else:
                if req.method in ("POST", "PUT", "PATCH"):
                    req.set_header_if_not("Content-Length", 0)
        else:
            req.set_header_if_not("Content-Type", "text/plain")
            req.set_header_if_not("Content-Length", req.body.size())

        if !basic_auth_password.isEmpty() or !basic_auth_username.isEmpty():
            auth_header = f"Basic {base64_encode(basic_auth_username + ":" basic_auth_password)}"
            req.set_header_if_not("Proxy-Authorization" if is_proxy else "Authorization", auth_header)

        if !bearer_token.isEmpty():
            auth_header = f"Bearer {bearer_token}"
            req.set_header_if_not("Proxy-Authorization" if is_proxy else "Authorization", auth_header)
        
        if !bearer_token_proxy.isEmpty():
            auth_header = f"Bearer {bearer_token_proxy}"
            req.set_header_if_not("Proxy-Authorization", auth_header)

        path = url_encode(req.path) if decode_url else req.path
        stream = io.StringIO()
        stream.write(f"{req.method} {req.path} HTTP/1.1\r\n")
        for k, v in req.headers.items():
            stream.write(f"{k}: {v}\r\n")
        
        strm.write(stream.getvalue()) # actually writes to socket

        # body
        if req.body.isEmpty():
            return write_content_with_provider(strm, req)
    
        if req.body.isEmpty():
            return strm.write(req.body)
        
        return True


    def process_request(strm, req, resp, close_connection):


    def handle_request(req, resp, close_connection):
        sock_stream = socket_stream(sock, timeouts...)
        if req.path.empty():
            raise error
        
        if !ssl:
            req.path = "http://" + host_and_port + req.path
            process_request(strm, req, res, close_connection)
        else:
            process_request(strm, req, res, close_connection)

def write_content_chunked(strm, content_provider, is_shutting_down, compressor):
    