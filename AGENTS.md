# AGENTS.md

You must respond in Japanese.
Generate code by referencing the actual code of the Open WebUI core and its related libraries, and adjust accordingly. Never write code based on speculation.
The Open WebUI core code is included as a submodule within the references folder.
Make sure to update it to the latest version before referencing.
Assume that all functions within the Tools class will be called by the AI. In particular, write function documentation comments that are specific and detailed enough for the AI to understand when to call them and what arguments to pass.
Do not include any functions in the Tools class that should not be called by the AI. (Starting with an underscore does not exempt them; internal methods should not be included in the Tools class.)
Also, prepare for potential errors due to arguments passed by the AI by implementing appropriate error handling within the functions and returning specific correction methods as return values.