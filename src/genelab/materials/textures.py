"""Texture wrappers (``gs.textures.*``) used by surface cfgs."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TextureCfg:
    """Base for Genesis texture configs."""

    def build(self, gs: Any) -> Any:
        raise NotImplementedError


@dataclass
class ColorTextureCfg(TextureCfg):
    """``gs.textures.ColorTexture`` — a flat RGB color."""

    color: tuple[float, ...] | None = None

    def build(self, gs: Any) -> Any:
        kwargs = {} if self.color is None else {"color": self.color}
        return gs.textures.ColorTexture(**kwargs)


@dataclass
class ImageTextureCfg(TextureCfg):
    """``gs.textures.ImageTexture`` — an image map loaded from ``image_path``."""

    image_path: str | None = None
    encoding: Literal["srgb", "linear"] | None = None

    def build(self, gs: Any) -> Any:
        kwargs: dict[str, Any] = {}
        if self.image_path is not None:
            kwargs["image_path"] = self.image_path
        if self.encoding is not None:
            kwargs["encoding"] = self.encoding
        return gs.textures.ImageTexture(**kwargs)
