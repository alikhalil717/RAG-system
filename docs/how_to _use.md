# How to Use

1. Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Place your PDF documents in the `data/pdfs` directory.

3. Start the application runtime from Python or from the notebook. The notebook in `app/src/test.ipynb` is the easiest interactive option.

4. Initialize the system by calling:

```python
from app import main

main.run_app(rebuild_index=True, kill_existing_reranker=True)
```

5. Ask a question with:

```python
answer = main.get_answer("What is the document about?", top_k=5, session_id="default")
print(answer)
```

6. When you finish, close the runtime with:

```python
main.close_app()
```

7. Use `rebuild_index=False` on later runs if you do not want to recreate the collection.
###
use the test notebook in src/test.ipynb for an interactive experience, or call the functions directly from Python for more control.