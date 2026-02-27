import base64
import io
import os
from dataclasses import dataclass
from typing import Any

import requests
import torch
from PIL.Image import Image as PILImage
from ezmm import MultimodalSequence
from ezmm.common.registry import item_registry
from transformers import AutoProcessor, AutoModel

from defame.common import Action, Results, logger
from defame.common.structured_logger import StructuredLogger
from defame.evidence_retrieval.tools.tool import Tool


class Geolocate(Action):
    """Performs geolocation to determine the country where an image was taken."""
    name = "geolocate"
    requires_image = True

    def __init__(self, image: str, top_k: int = 5):
        """
        @param image: The reference of the image to be geolocated.
        @param top_k: Number of candidates to include in the list of
            most likely countries.
        """
        self._save_parameters(locals())
        self.image = item_registry.get(reference=image)
        self.top_k = top_k

    def __eq__(self, other):
        return isinstance(other, Geolocate) and self.image == other.image

    def __hash__(self):
        return hash((self.name, self.image))


@dataclass
class GeolocationResults(Results):
    text: str
    most_likely_location: str
    top_k_locations: list[str]
    model_output: Any | None = None

    def __str__(self):
        locations_str = ', '.join(self.top_k_locations)
        text = (f'Most likely location: {self.most_likely_location}\n'
                f'Top {len(self.top_k_locations)} locations: {locations_str}')
        return text

    def is_useful(self) -> bool | None:
        return self.model_output is not None


class Geolocator(Tool):
    """Localizes a given photo."""
    name = "geolocator"
    actions = [Geolocate]
    summarize = False

    def __init__(self, model_name: str = "geolocal/StreetCLIP", top_k=10,
                 server_url: str = "http://localhost:5555", **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.server_url = os.environ.get("GEOLOCATOR_URL", server_url)

        if server_url:
            logger.log(f"Geolocator using remote server at {server_url}")
            self.processor = None
            self.model = None
        else:
            logger.log("Initializing geolocator...")
            self.model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            self.device = torch.device(self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

            if self.device.type == 'cuda':
                free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
                logger.log(f"GPU memory: {free_mem / 1e9:.1f}GB free / {total_mem / 1e9:.1f}GB total. "
                           f"Model size: {model_size / 1e6:.0f}MB")

            self.model.to(self.device)

    def _perform(self, action: Geolocate, structured_logger: StructuredLogger | None = None) -> Results:
        return self.locate(action.image.image)

    def locate(self, image: PILImage, choices: list[str] | None = None) -> GeolocationResults:
        """
        Perform geolocation on an image.

        :param image: A PIL image.
        :param choices: A list of location choices. If None, uses a default list of countries.
        :return: A GeoLocationResult object containing location predictions and their probabilities.
        """
        if choices is None:
            choices = ['Albania', 'Andorra', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Bermuda',
                       'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China',
                       'Colombia', 'Croatia', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Estonia',
                       'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guam', 'Guatemala', 'Hungary',
                       'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kenya',
                       'Kyrgyzstan', 'Laos', 'Latvia', 'Lesotho', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar',
                       'Malaysia', 'Malta', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Netherlands', 'New Zealand',
                       'Nigeria', 'Norway', 'Pakistan', 'Palestine', 'Peru', 'Philippines', 'Poland', 'Portugal',
                       'Puerto Rico', 'Romania', 'Russia', 'Rwanda', 'Senegal', 'Serbia', 'Singapore', 'Slovakia',
                       'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Swaziland', 'Sweden',
                       'Switzerland', 'Taiwan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine',
                       'United Arab Emirates',
                       'United Kingdom', 'United States', 'Uruguay']

        if self.server_url:
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            payload = {"image_b64": image_b64, "top_k": self.top_k}
            if choices is not None:
                payload["choices"] = choices
            response = requests.post(f"{self.server_url}/geolocate", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            result = GeolocationResults(
                text=data["text"],
                most_likely_location=data["most_likely_location"],
                top_k_locations=data["top_k_locations"],
            )
        else:
            inputs = self.processor(text=choices, images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            prediction = logits_per_image.softmax(dim=1)

            confidences = {choices[i]: round(float(prediction[0][i].item()), 2) for i in range(len(choices))}
            top_k_locations = dict(sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:self.top_k])
            most_likely_location = max(top_k_locations, key=top_k_locations.get)
            model_output = logits_per_image.detach().cpu()
            result = GeolocationResults(
                text=f"The most likely countries where the image was taken are: {top_k_locations}",
                most_likely_location=most_likely_location,
                top_k_locations=list(top_k_locations.keys()),
                model_output=model_output
            )

        logger.log(str(result))
        return result

    def _summarize(self, result: GeolocationResults, **kwargs) -> MultimodalSequence | None:
        return MultimodalSequence(result.text)  # TODO: Improve summary w.r.t. uncertainty
