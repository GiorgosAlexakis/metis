import argparse

from .server import mcp
from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult


def main():
    """MCP Metis command-line interface."""
    parser = argparse.ArgumentParser(
        description="Gives you the ability to read Wikipedia articles and convert them to Markdown."
    )
    parser.parse_args()
    mcp.run()


if __name__ == "__main__":
    main()
