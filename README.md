# genai - image

Most research papers present mathematical equations—sometimes distilled to just one crucial formula—whose effectiveness is evaluated across various datasets and compared against existing methods. While authors typically release their code, navigating through the boilerplate can be challenging. This repository implements foundational architectures and incorporates improvements based on research across different image generative AI tasks. By presenting code alongside equations, it aims to clarify how concepts integrate within the broader framework, enhancing understanding of the underlying principles.

This is mostly my personal notes. 


## setup using uv 
- first install asdf 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.11                   

uv python pin 3.11.13

uv venv

source .venv/bin/activate

uv pip install -r requirements.txt

uv pip install package1 package2 package3

# Compile requirements.txt from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt

# Sync your environment with requirements.txt
uv pip sync requirements.txt


## update all packages 
uv pip install --upgrade $(uv pip list --outdated | cut -d' ' -f1)
```

## 