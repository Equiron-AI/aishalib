import logging
import re


logger = logging.getLogger(__name__)


def parseToolResponse(response, tool_names):
    regex_pattern = r"(" + "|".join(item + ":" for item in tool_names) + ")"
    tools = re.split(regex_pattern, response)
    tools = [tool.strip() for tool in tools if tool.strip()]
    tools_dict = {}
    for i in range(1, len(tools), 2):
        tool_name = tools[i-1].strip(':').strip()
        tool_argument = tools[i].strip()
        tools_dict[tool_name] = tool_argument
        logger.info(tool_name + ": " + tool_argument)
    return tools_dict