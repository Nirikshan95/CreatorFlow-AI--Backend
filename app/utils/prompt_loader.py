from pathlib import Path
from typing import Optional


class PromptLoader:
    def __init__(self, prompts_dir: str = "app/prompts"):
        self.prompts_dir = Path(prompts_dir)
    
    def load_prompt(self, prompt_name: str) -> Optional[str]:
        prompt_file = self.prompts_dir / f"{prompt_name}.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return None
    
    def list_prompts(self) -> list[str]:
        return [f.stem for f in self.prompts_dir.glob("*.txt")]


prompt_loader = PromptLoader()
