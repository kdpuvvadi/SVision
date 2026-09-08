from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraStatus:
    mode: str
    connected: bool
    message: str


class ImageSource:
    """Current source is uploaded files. GigE/GenICam is reserved for later."""

    def __init__(self, gige_ip: str = "") -> None:
        self.gige_ip = gige_ip

    def status(self) -> CameraStatus:
        if self.gige_ip:
            return CameraStatus(
                mode="upload",
                connected=False,
                message=f"GigE camera {self.gige_ip} is saved but not connected. Inspection uses uploaded images.",
            )
        return CameraStatus(
            mode="upload",
            connected=False,
            message="No camera connected. Upload images to inspect. GenICam/GigE can be added later.",
        )

    def grab(self):
        raise RuntimeError("Camera grab is not available. Upload an image to inspect.")
