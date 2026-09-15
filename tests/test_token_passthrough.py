"""
Tests for API token passthrough: without a static SEMAPHORE_API_TOKEN, the bearer
token sent by the MCP client on the HTTP transport is forwarded to SemaphoreUI.
A configured static token always wins and client tokens are ignored.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from semaphore_mcp.api import SemaphoreAPIClient, create_client, parse_bearer_token
from semaphore_mcp.server import SemaphoreMCPServer

BASE_URL = "http://test.example.com"
NO_STATIC_TOKEN = {"SEMAPHORE_API_TOKEN": ""}


def _ok_response(payload):
    response = MagicMock()
    response.status_code = 200
    response.content = b"[]"
    response.text = ""
    response.json.return_value = payload
    return response


def _mcp_context(request):
    """Build a fake FastMCP Context whose request_context carries `request`."""
    ctx = MagicMock()
    ctx.request_context.request = request
    return ctx


def _http_request(authorization=None):
    request = MagicMock()
    request.headers = {}
    if authorization is not None:
        request.headers["authorization"] = authorization
    return request


def _client_without_static_token(token_provider):
    with patch.dict("os.environ", NO_STATIC_TOKEN):
        return SemaphoreAPIClient(BASE_URL, token_provider=token_provider)


def _server_without_static_token():
    with patch.dict("os.environ", NO_STATIC_TOKEN):
        return SemaphoreMCPServer(BASE_URL, "")


class TestParseBearerToken:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Bearer abc123", "abc123"),
            ("bearer abc123", "abc123"),
            ("  Bearer   abc123  ", "abc123"),
            ("Bearer", None),
            ("Bearer   ", None),
            ("Basic dXNlcjpwYXNz", None),
            ("abc123", None),
            ("", None),
            (None, None),
        ],
    )
    def test_parse(self, value, expected):
        assert parse_bearer_token(value) == expected


class TestClientTokenResolution:
    def test_provider_token_used_without_static_token(self):
        client = _client_without_static_token(lambda: "client-token")
        with patch.object(
            client.session, "request", return_value=_ok_response([])
        ) as request:
            client.list_projects()

        assert "Authorization" not in client.session.headers
        headers = request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer client-token"

    def test_static_token_wins_and_provider_is_ignored(self):
        provider = MagicMock(return_value="client-token")
        client = SemaphoreAPIClient(
            BASE_URL, token="static-token", token_provider=provider
        )
        with patch.object(
            client.session, "request", return_value=_ok_response([])
        ) as request:
            client.list_projects()

        provider.assert_not_called()
        assert client.session.headers["Authorization"] == "Bearer static-token"
        assert "headers" not in request.call_args.kwargs

    def test_no_token_at_all_sends_no_authorization(self):
        client = _client_without_static_token(lambda: None)
        with patch.object(
            client.session, "request", return_value=_ok_response([])
        ) as request:
            client.list_projects()

        assert "Authorization" not in client.session.headers
        assert "headers" not in request.call_args.kwargs

    def test_static_token_without_provider_is_unchanged(self):
        client = SemaphoreAPIClient(BASE_URL, token="static-token")
        with patch.object(
            client.session, "request", return_value=_ok_response([])
        ) as request:
            client.list_projects()

        assert client.session.headers["Authorization"] == "Bearer static-token"
        request.assert_called_once_with("GET", f"{BASE_URL}/api/projects", timeout=30.0)

    def test_provider_is_consulted_on_every_request(self):
        tokens = iter(["first", "second"])
        client = _client_without_static_token(lambda: next(tokens))
        with patch.object(
            client.session, "request", return_value=_ok_response([])
        ) as request:
            client.list_projects()
            client.list_projects()

        sent = [c.kwargs["headers"]["Authorization"] for c in request.call_args_list]
        assert sent == ["Bearer first", "Bearer second"]

    def test_explicit_headers_are_preserved(self):
        client = _client_without_static_token(lambda: "client-token")
        with patch.object(
            client.session, "request", return_value=_ok_response({})
        ) as request:
            client._request("GET", "ping", headers={"X-Custom": "1"})

        headers = request.call_args.kwargs["headers"]
        assert headers["X-Custom"] == "1"
        assert headers["Authorization"] == "Bearer client-token"

    def test_raw_output_uses_provider_token(self):
        client = _client_without_static_token(lambda: "client-token")
        response = MagicMock()
        response.text = "raw output"
        with patch.object(client.session, "request", return_value=response) as request:
            assert client.get_task_raw_output(1, 2) == "raw output"

        headers = request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer client-token"

    def test_raw_output_with_static_token_adds_no_headers(self):
        client = SemaphoreAPIClient(
            BASE_URL, token="static-token", token_provider=lambda: "client-token"
        )
        response = MagicMock()
        response.text = "raw output"
        with patch.object(client.session, "request", return_value=response) as request:
            assert client.get_task_raw_output(1, 2) == "raw output"

        request.assert_called_once_with(
            "GET", f"{BASE_URL}/api/project/1/tasks/2/raw_output", timeout=30.0
        )

    def test_create_client_accepts_provider(self):
        provider = MagicMock(return_value="client-token")
        client = create_client(BASE_URL, "static-token", token_provider=provider)
        assert client.token_provider is provider


class TestServerRequestToken:
    def test_client_uses_server_request_token(self):
        server = SemaphoreMCPServer(BASE_URL, "static-token")
        assert server.semaphore.token_provider == server.get_request_token

    def test_reads_bearer_header_from_current_request(self):
        server = SemaphoreMCPServer(BASE_URL, "static-token")
        ctx = _mcp_context(_http_request("Bearer client-token"))
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server.get_request_token() == "client-token"

    def test_returns_none_without_authorization_header(self):
        server = SemaphoreMCPServer(BASE_URL, "static-token")
        ctx = _mcp_context(_http_request())
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server.get_request_token() is None

    def test_returns_none_on_stdio_transport(self):
        # On stdio the request context carries no HTTP request.
        server = SemaphoreMCPServer(BASE_URL, "static-token")
        ctx = _mcp_context(None)
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server.get_request_token() is None

    def test_returns_none_outside_of_a_request(self):
        server = SemaphoreMCPServer(BASE_URL, "static-token")
        ctx = MagicMock()
        type(ctx).request_context = PropertyMock(
            side_effect=ValueError("Context is not available outside of a request")
        )
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server.get_request_token() is None

    def test_returns_none_without_active_context(self):
        # Outside a tool call FastMCP has no request context at all.
        server = SemaphoreMCPServer(BASE_URL, "static-token")
        assert server.get_request_token() is None

    def test_header_token_reaches_semaphore_api_without_static_token(self):
        server = _server_without_static_token()
        ctx = _mcp_context(_http_request("Bearer client-token"))
        with patch.object(server.mcp, "get_context", return_value=ctx):
            with patch.object(
                server.semaphore.session, "request", return_value=_ok_response([])
            ) as request:
                server.semaphore.list_projects()

        headers = request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer client-token"

    def test_static_token_ignores_client_header(self):
        server = SemaphoreMCPServer(BASE_URL, "static-token")
        ctx = _mcp_context(_http_request("Bearer client-token"))
        with patch.object(server.mcp, "get_context", return_value=ctx):
            with patch.object(
                server.semaphore.session, "request", return_value=_ok_response([])
            ) as request:
                server.semaphore.list_projects()

        assert "headers" not in request.call_args.kwargs
        assert (
            server.semaphore.session.headers["Authorization"] == "Bearer static-token"
        )

    def test_run_logs_passthrough_mode_without_static_token(self, caplog):
        server = _server_without_static_token()
        with patch.object(server.mcp, "run"):
            with caplog.at_level("INFO", logger="semaphore_mcp"):
                server.run(transport="http")
        assert "forwarding the Authorization: Bearer token" in caplog.text

    def test_run_warns_without_token_on_stdio(self, caplog):
        server = _server_without_static_token()
        with patch.object(server.mcp, "run"):
            with caplog.at_level("WARNING", logger="semaphore_mcp"):
                server.run(transport="stdio")
        assert "stdio transport carries no HTTP headers" in caplog.text
