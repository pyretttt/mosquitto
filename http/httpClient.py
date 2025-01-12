import requests
import functools
import json
import textwrap


format_json = functools.partial(json.dumps, indent=2, sort_keys=True)
indent = functools.partial(textwrap.indent, prefix='  ')


def format_response(resp):
    """Pretty-format 'requests.Response'"""
    headers = '\n'.join(f'{k}: {v}' for k, v in resp.headers.items())
    content_type = resp.headers.get('Content-Type', '')
    if 'application/json' in content_type:
        try:
            body = format_json(resp.json())
        except json.JSONDecodeError:
            body = resp.text
    else:
        body = resp.text
    s = textwrap.dedent("""
    RESPONSE
    ========
    status_code: {status_code}
    headers:
    {headers}
    body:
    {body}
    ========
    """).strip()

    s = s.format(
        status_code=resp.status_code,
        headers=indent(headers),
        body=indent(body),
    )
    return s


if __name__ == "__main__":
    r = requests.get("http://127.0.0.1:3000/home")
    
    print(format_response(r))