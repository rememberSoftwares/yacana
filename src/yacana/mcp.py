import copy
import json
import logging
import requests
import uuid
from typing import Dict, List, Any, Optional

from .exceptions import McpBadToolConfig, McpBadTransport, McpResponseError, McpServerNotYetInitialized
from .tool import Tool, ToolType


class Mcp:
    """
    Connects to an HTTP streamable MCP server to get the tools and make them available to your Tasks.

    Parameters
    ----------
    server_url: str
        Url of the remote MCP server.
    headers: dict
        Dict of headers passed to all requests made to the MCP server. Use this for authentification.

    Attributes
    ----------
    server_url: str
        Url of the remote MCP server.
    headers: dict
        Dict of headers passed to all requests made to the MCP server. Use this for authentification.
    session: request.session
        The network session from connecting to the server using request.
    tools: List[Tool]
        The list of tools auto discovered on the server.
    initialized: bool
        Is the connection to the server established.
    session_id: Optional[str]
        Id of the request session.
    """

    def __init__(self, server_url: str, headers: dict = None) -> None:
        self.server_url = server_url.rstrip('/')
        self.headers = headers if headers is not None else {}
        self.session = requests.Session()
        self.tools: List[Tool] = []
        self.initialized = False
        self.session_id: Optional[str] = None

    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Makes a JSON-RPC request to the MCP server and parses the response as JSON or SSE (Server-Sent Events).

        Parameters
        ----------
        method : str
            The JSON-RPC method to call.
        params : Dict[str, Any], optional
            The parameters to pass to the method. Defaults to None.

        Returns
        -------
        Dict[str, Any]
            The result of the method call.
        """
        request_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers
        }

        # Add session ID header if we have one
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        try:
            response = self.session.post(
                self.server_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            # Check if server returned a session ID during initialization
            if method == "initialize" and "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
                logging.debug(f"Server assigned session ID: {self.session_id}")

            # Handle different response types
            content_type = response.headers.get('content-type', '').lower()

            if 'application/json' in content_type:
                result = response.json()
                logging.debug(f"Getting result from MCP server with protocol: application/json. Response = {result}")
                if "error" in result:
                    raise McpResponseError(f"MCP Error: {result['error']}")
                return result.get("result", {})

            elif 'text/event-stream' in content_type:
                # Handle SSE response - for simplicity, we'll read the first JSON response
                # In a production client, you'd want to properly handle the SSE stream
                logging.debug(f"Getting result from MCP server with protocol: SSE. Response = {response}")
                return self._handle_sse_response(response)

            else:
                # If no specific content type, try to parse as JSON
                try:
                    logging.warning("Received unexpected content type, trying to parse as JSON")
                    result = response.json()
                    logging.debug(f"Getting result from MCP server with unknown protocol. Response = {result}")
                    if "error" in result:
                        logging.error(f"MCP Error: {result['error']}")
                        raise McpResponseError(f"MCP Error: {result['error']}")
                    return result.get("result", {})
                except:
                    raise McpBadTransport(f"Unexpected response format: {response.text}")

        except requests.exceptions.HTTPError as e:
            raise McpBadTransport(f"HTTP Error {e.response.status_code}: {e.response.text}")

    def _handle_sse_response(self, response) -> Dict[str, Any]:
        """
        Handle Server-Sent Events (SSE) response.
        Reads the streamed response line by line and reconstructs SSE events properly.

        Parameters
        ----------
        response : requests.Response
            Streamed response from the server (with text/event-stream content-type).

        Returns
        -------
        Dict[str, Any]
            The first valid 'result' found in the SSE stream.
        """
        event_data_lines = []

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue

            line = line.strip()
            # Comment line or empty
            if line == '':
                # End of one event
                if event_data_lines:
                    full_data = '\n'.join(event_data_lines)
                    try:
                        data = json.loads(full_data)
                        if "result" in data:
                            #print("=> ", data["result"])
                            return data["result"]
                        elif "error" in data:
                            raise McpResponseError(f"MCP Error: {data['error']}")
                    except json.JSONDecodeError:
                        pass  # Ignore and continue
                    event_data_lines = []  # Reset buffer for next event
                continue

            if line.startswith('data:'):
                #print("strip => ", line[5:].lstrip())
                # @todo Could be just 'data:' (empty), so use slicing carefully
                event_data_lines.append(line[5:].lstrip())

        raise McpBadTransport("Invalid SSE stream format")

    def connect(self) -> bool:
        """
        Initializes connection with the MCP server
        """
        logging.info(f"[MCP] Connecting to MCP server ({self.server_url})...")
        # Initialize the protocol
        init_result = self._make_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "yacana-mcp-client",
                "version": "0.1.0"
            }
        })

        server_info = init_result.get('serverInfo', {})
        logging.info(f"[MCP] Connected to MCP server: {server_info.get('name', 'Unknown')} v{server_info.get('version', 'Unknown')}")

        # List available tools
        tools_result = self._make_request("tools/list")
        tools = tools_result.get("tools", [])

        for tool_info in tools:
            try:
                tool = Tool(tool_info.get("name"),
                            tool_info.get("description"),
                            function_ref=self._call_tool,
                            mcp_input_schema=tool_info["inputSchema"],
                            optional=True)
                self.tools.append(tool)
                logging.info(f"[MCP] Available tool: {tool.tool_name} - {tool.function_description}")
            except McpBadToolConfig as e:
                logging.warning(f"[MCP] Tool will be excluded due to : {e.message}")

        if not tools:
            logging.info("[MCP] No tools available on this server")

        self.initialized = True
        return True

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls a tool on the MCP server using request.

        Parameters
        ----------
        tool_name : str
            The name of the tool to call.
        arguments : Dict[str, Any]
            The arguments to pass to the tool.

        Returns
        -------
        Dict[str, Any]
            The result of the tool call.
        """
        if not self.initialized:
            raise McpServerNotYetInitialized("Cannot call tool on server. Client not initialized")
        result = self._make_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        return result

    def get_tools_as(self, tools_type: ToolType, optional=None) -> List[Tool]:
        """
        Returns the tools from the remote MCP server as a list of Tool objects. You choose how these tools will be called
        by specifying the tools_type parameter. This list of tools must be given to a Task() object.
        !Warning!: This returns a deep copy of the tools list !

        Parameters
        ----------
        tools_type : ToolType
            The type of tools to return.
        optional : bool, optional
            Changes the optional status of all tools in the MCP client. !WARNING! The status will remain for the next Tasks!

        Returns
        -------
        List[Tool]
            A list of Tool objects of the specified type.
        """

        if not self.initialized:
            raise McpServerNotYetInitialized("Cannot get tools from server. Client not initialized")
        tools_copy = copy.deepcopy(self.tools)
        for tool in tools_copy:
            tool.tool_type = tools_type
            if optional is not None:
                tool.optional = optional
        return tools_copy

    def forget_tool(self, tool_name: str) -> None:
        """
        Forgets about a tool from the MCP server. Note that using a tool leaves a trace in the history, hence it is not
        possible to completely forget about a tool. However, the tool won't be proposed for future tasks. Call this method
        right after connecting to the MCP server.

        Parameters
        ----------
        tool_name : str
            The name of the tool to delete.
        """
        if not self.initialized:
            raise McpServerNotYetInitialized("Cannot forget tool. Client not initialized")
        self.tools = [tool for tool in self.tools if tool.tool_name != tool_name]
        logging.info(f"Tool '{tool_name}' deleted from MCP client.")

    def disconnect(self):
        """
        Explicitly disconnects from the server
        """
        if self.session_id:
            try:
                headers = {"Mcp-Session-Id": self.session_id, **self.headers}
                self.session.delete(self.server_url, headers=headers, timeout=5)
            except:
                pass  # Ignore errors during cleanup
        self.session.close()
