import os
from functools import partial

import requests
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData
from requests.exceptions import RequestException

from agent import APIProvider, sampling_loop

mcp = FastMCP("metis")

http_logs = []

def _api_response_callback(request, response, error, tab, response_state):
    if error:
        tab.append(f"Error: {error}")
    else:
        tab.append(response.text if hasattr(response, "text") else str(response))
    response_state.append(response)


@mcp.tool()
async def perform_platform_action(user_message: str) -> str:
    """
    Based on the user's message and additional provided instructions on how to use the platform performs on a target platform.

    Usage:
        perform_platform_action("How do I create an organization in stytch?")
    """
    try:
        instructions = """
            You are an agent for stytch (an all-in-one platform for modern authentication) that helps new users of a SaaS platform to perform actions on stytch.
            1. Navigate to https://stytch.com/.
            2. Click "Get Started".
            3. Click "Continue with Google".
            4. Click On the First Account.
            5. Click "Continue".
            6. Click "Create an organization".
            7. Click the "First name" field.
            8. Type "John tab Doe tab Metis tab Multimodal AI Agent Hackathon".
            9. Click the "Consumer AuthenticationIntegrating auth into an application designed.
            10. Click "Next".
            11. Click "Get started".
        """
        messages: list[BetaMessageParam] = [
            BetaMessageParam(content=BetaTextBlockParam(text=user_message), role="user"),
        ]
        # run the agent sampling loop with the newest message
        response_state = []
        response: list[BetaMessageParam]=  await sampling_loop(
            system_prompt_suffix=instructions,
            model="claude-3-5-sonnet-20241022",
            provider=APIProvider.ANTHROPIC,
            messages=messages,
            api_response_callback=partial(
                _api_response_callback,
                tab=http_logs,
                response_state=response_state,
            ),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            only_n_most_recent_images=5,
        )
        if len(response) == 0:
            return "Please try again."
        return response[0].content.text
    except ValueError as e:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error: {str(e)}")) from e
