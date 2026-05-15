
The project uses DeepSeek Chat through the OpenAI-compatible ChatOpenAI client.
## How it is used

- Retrieved chunks are inserted into a prompt together with the user question and recent chat history.
- The model is instructed to answer briefly and to say `I don't know` when the context is insufficient.
- Generation is the only step that currently depends on an external API key.



## how to choose the best  model

- DeepSeek was chosen for its strong performance on short answer generation and its compatibility with the existing LangChain/OpenAI interface.
-if the prohjrct needed a more powerful model for long-form generation, I would consider alternatives like GPT-4 or Claude 2, but for the current scope, DeepSeek provides a good balance of quality and cost.
-and becouse of the high cost of claude 2 ,when we are going to use this project for code generation ,i will consider using a code-specialized model like DeepSeek code or gpt-4 with code capabilities, as they are better at understanding and generating programming language context

