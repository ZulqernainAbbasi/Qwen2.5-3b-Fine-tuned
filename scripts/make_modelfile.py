"""Write the Ollama Modelfile for car-assistant, reusing qwen2.5's tool-capable chat TEMPLATE.

Run on Windows: python -m scripts.make_modelfile ; then: ollama create car-assistant -f Modelfile -q q4_K_M
"""
import subprocess

from finetune.build_dataset import SYSTEM_PROMPT

BASE = "qwen2.5:3b-instruct"


def main() -> None:
    template = subprocess.run(["ollama", "show", BASE, "--template"], capture_output=True, text=True,
                              check=True, encoding="utf-8").stdout.strip()
    content = (
        "FROM ./outputs/merged\n"
        f'TEMPLATE """{template}"""\n'
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
        "PARAMETER temperature 0.3\n"
        "PARAMETER num_ctx 4096\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n'
    )
    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(content)
    print("Modelfile written")


if __name__ == "__main__":
    main()
