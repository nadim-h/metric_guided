# Purpose
Act as an expert software engineer specialized in {language} code.

# Task
Refactor the code in `{filename}` to improve maintainability and code quality without altering semantic behaviour.

# Requirements
Strictly follow these requirements:
- Use the `save` tool to overwrite `{filename}` with the full refactored content, do not introduce any other new files.
- Do not edit the `{filename}` name.
- Do not alter the program's logic or output behavior.
- Do not introduce new libraries, tools, packages or dependencies.
- Do not perform general fixes or modifications, outside of remediating the maintainability issues.

# Execution
Now proceed with the refactoring of `{filename}` by:
1. Using the `read` tool to understand the content of `{filename}`.
2. Review and analyze the code and identify maintainability issues.
3. Plan refactoring steps to remediate the identified issues.
4. Execute the refactoring steps and overwrite `{filename}` with the full refactored program using the `save` tool.
5. Use the `read` tool to understand the new content of `{filename}`.
6. Validate and analyze that all maintainability issues are remediated and no new ones are found.
7. If new issues are found, go back to step 3, otherwise terminate.

