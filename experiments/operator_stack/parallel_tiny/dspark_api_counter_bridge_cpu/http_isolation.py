"""ASGI route isolation and successful body publication observation; stdlib only."""
import json

ALLOWED = {
    ('GET', '/private/dspark/identity'),
    ('POST', '/private/dspark/completions'),
    ('POST', '/private/dspark/finish'),
}


class PrivateOnlyASGI:
    def __init__(self, app, bridge):
        self.app = app
        self.bridge = bridge

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            return await self.app(scope, receive, send)
        if scope['type'] != 'http':
            if scope['type'] == 'websocket':
                await send({'type': 'websocket.close', 'code': 1008})
                return
            raise ValueError('Unexpected ASGI scope')
        if (scope['method'], scope['path']) not in ALLOWED:
            body = json.dumps({'error': 'This private service accepts only the fixed DSpark protocol'}).encode()
            await send({'type': 'http.response.start', 'status': 403,
                'headers': [(b'content-type', b'application/json')]})
            await send({'type': 'http.response.body', 'body': body})
            return

        async def observed_send(message):
            window_id = scope.get('dspark_api_completion_window')
            try:
                if window_id is not None and message['type'] == 'http.response.start':
                    self.bridge.note_http_send_begin(window_id, message['status'])
                await send(message)
                if window_id is not None and message['type'] == 'http.response.body':
                    self.bridge.note_http_body(window_id, message.get('body', b''),
                        not message.get('more_body', False))
            except BaseException as exc:
                if window_id is not None:
                    self.bridge.fail('http_output_send', exc)
                raise

        return await self.app(scope, receive, observed_send)
