from openai import OpenAI
import openai
import tiktoken
import numpy as np
from transformers import Pipeline

from defame.common.medium import Image
from config.globals import api_keys
from defame.common.prompt import Prompt
from defame.common.modeling.model import Model
from defame.common.logger import logger


class OpenAIAPI:
    def __init__(self, model: str):
        self.model = model
        if not api_keys["openai_api_key"]:
            raise ValueError("No OpenAI API key provided. Add it to config/api_keys.yaml")
        self.client = OpenAI(api_key=api_keys["openai_api_key"])

    def __call__(self, prompt: Prompt, system_prompt: str, max_prompt_tokens: int = None, encoding=None, **kwargs):
        if prompt.has_videos():
            raise ValueError(f"{self.model} does not support videos.")

        if prompt.has_audios():
            raise ValueError(f"{self.model} does not support audios.")

        content = format_for_gpt(prompt, encoding=encoding, max_tokens=max_prompt_tokens)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message.content
    

class GPTModel(Model):
    open_source = False
    encoding = tiktoken.get_encoding("cl100k_base")
    accepts_images = True
    accepts_videos = False
    accepts_audio = False

    def load(self, model_name: str) -> Pipeline | OpenAIAPI:
        return OpenAIAPI(model=model_name)

    def _generate(
        self,
        prompt: Prompt,
        temperature: float,
        top_p: float,
        top_k: int,
        system_prompt: Prompt | None = None
    ) -> str:
        kwargs = {"temperature": temperature, "top_p": top_p}
        if self.name in ["gpt_5"]:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)

        n_system_tokens = self.count_tokens(system_prompt) if system_prompt else 0
        n_prompt_tokens = self.count_tokens(prompt)
        n_total_tokens = n_system_tokens + n_prompt_tokens
        
        max_prompt_tokens = None
        if n_total_tokens + self.max_response_len > self.context_window:
            max_prompt_tokens = self.context_window - self.max_response_len - n_system_tokens
            logger.warning(f"Input too long ({n_total_tokens} tokens), pruning prompt to {max_prompt_tokens} tokens.")

        try:
            return self.api(
                prompt,
                system_prompt=system_prompt,
                max_prompt_tokens=max_prompt_tokens,
                encoding=self.encoding,
                **kwargs,
            )
        except openai.RateLimitError as e:
            logger.critical(f"OpenAI rate limit hit!")
            logger.critical(repr(e))
            quit()
        except openai.AuthenticationError as e:
            logger.critical(f"Authentication at OpenAI API was unsuccessful!")
            logger.critical(e)
            quit()
        except Exception as e:
            logger.warning("Error while calling the LLM! Continuing with empty response.\n" + str(e))
            logger.warning("Prompt used:\n" + str(prompt))
        return ""

    def count_tokens(self, prompt: Prompt | str) -> int:
        n_text_tokens = len(self.encoding.encode(str(prompt)))
        n_image_tokens = 0
        if isinstance(prompt, Prompt) and prompt.has_images():
            for image in prompt.images:
                n_image_tokens += self.count_image_tokens(image)
        return n_text_tokens + n_image_tokens

    def count_image_tokens(self, image: Image):
        """See the formula here: https://openai.com/api/pricing/"""
        n_tiles = np.ceil(image.width / 512) * np.ceil(image.height / 512)
        return 85 + 170 * n_tiles
    

def format_for_gpt(prompt: Prompt, encoding=None, max_tokens: int = None):
    content_formatted = []
    remaining = max_tokens  # None means no limit

    for block in prompt.to_list():
        if remaining is not None and remaining <= 0:
            break

        if isinstance(block, str):
            if encoding is not None and remaining is not None:
                tokens = encoding.encode(block)
                if len(tokens) > remaining:
                    block = encoding.decode(tokens[:remaining])
                    remaining = 0
                else:
                    remaining -= len(tokens)
            content_formatted.append({"type": "text", "text": block})

        elif isinstance(block, Image):
            if encoding is not None and remaining is not None:
                n_tiles = np.ceil(block.width / 512) * np.ceil(block.height / 512)
                image_tokens = 85 + 170 * n_tiles
                ref_tokens = len(encoding.encode(block.reference))
                if image_tokens + ref_tokens > remaining:
                    continue  # skip image if it doesn't fit
                remaining -= image_tokens + ref_tokens
            image_encoded = block.get_base64_encoded()
            content_formatted.append({"type": "text", "text": block.reference})
            content_formatted.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_encoded}"}
            })

    return content_formatted


if __name__ == "__main__":
    from defame.common.modeling import make_model
    model = make_model("gpt_5")
    prompt = Prompt(text="test")
    print(model.count_tokens(prompt))
    