# Purpose
Act as an expert software engineer specialized in {language} code.

# Task
Refactor the code in `{filename}` to improve maintainability and code quality, guided strictly by CodeScene's code health metric, without altering semantic behavior.

# Tools
The CodeScene MCP server is available as codescene. Use:
- code_health_review — analyze code health for a specific file.

# Requirements
Strictly follow these requirements:
- Use the `save` tool to overwrite `{filename}` with the full refactored content, do not introduce any other new files.
- Do not edit the `{filename}` name.
- Do not alter the program's logic or output behavior.
- Do not introduce, import, download, or install any new libraries, packages, tools, or dependencies. Use only standard libraries or existing imports.
- Do not make any changes, except those strictly necessary to remediate the identified maintainability issues.

# Execution
Now proceed with the refactoring of `{filename}` by:
1. Using the `read` tool to understand the content of `{filename}`.
2. Running code_health_review to identify concrete maintainability issues.
3. Planning refactoring steps to remediate the identified issues.
4. Executing the refactoring steps and overwriting `{filename}` with the full refactored program using the `save` tool.
5. Running code_health_review on the updated `{filename}`.
6. Validating that all identified maintainability issues are remediated and no new ones are introduced.
7. If new issues are found, go back to step 3, otherwise terminate.


