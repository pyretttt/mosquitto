from mitmproxy import http

def response(flow: http.HTTPFlow) -> None:
    # Check if the response matches your criteria (e.g., URL, content type)
    if "ya.ru" in flow.request.pretty_url and flow.response.content:
        # Modify the response content
        modified_content = "hui"
        flow.response.text = modified_content

        # You can also modify headers, status code, etc.
        flow.response.headers["X-Modified-By"] = "Mitmproxy Addon"
        # flow.response.status_code = 200