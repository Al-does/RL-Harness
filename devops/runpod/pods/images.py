"""Resolve public Docker Hub tags to immutable image digests."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request


class ImageResolutionError(RuntimeError):
    pass


def resolve_image_digest(image: str) -> tuple[str, str]:
    """Return (digest-pinned reference, sha256 digest).

    Automatic resolution is intentionally limited to public Docker Hub images.
    Private/custom registries must be supplied as an already digest-pinned
    reference so this tool never needs to persist registry credentials.
    """
    if "@sha256:" in image:
        digest = "sha256:" + image.rsplit("@sha256:", 1)[1]
        return image, digest

    reference = image
    if reference.startswith("docker.io/"):
        reference = reference[len("docker.io/") :]
    first = reference.split("/", 1)[0]
    if "." in first or ":" in first and "/" in reference:
        raise ImageResolutionError(
            "non-Docker-Hub images must be supplied by digest "
            "(registry/repository@sha256:...)"
        )
    if "/" not in reference:
        reference = f"library/{reference}"
    repository, separator, tag = reference.rpartition(":")
    if not separator or "/" in tag:
        repository, tag = reference, "latest"

    scope = f"repository:{repository}:pull"
    token_url = "https://auth.docker.io/token?" + urllib.parse.urlencode(
        {"service": "registry.docker.io", "scope": scope}
    )
    try:
        with urllib.request.urlopen(token_url, timeout=30) as response:
            token = json.load(response)["token"]
        request = urllib.request.Request(
            f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": ", ".join(
                    (
                        "application/vnd.oci.image.index.v1+json",
                        "application/vnd.docker.distribution.manifest.list.v2+json",
                        "application/vnd.oci.image.manifest.v1+json",
                        "application/vnd.docker.distribution.manifest.v2+json",
                    )
                ),
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read(1)
            digest = response.headers.get("Docker-Content-Digest")
    except Exception as error:  # noqa: BLE001
        raise ImageResolutionError(
            f"could not resolve public image digest ({type(error).__name__})"
        ) from None
    if not digest or not digest.startswith("sha256:"):
        raise ImageResolutionError(
            "Docker Hub response omitted Docker-Content-Digest"
        )
    original_repository = image.rsplit(":", 1)[0]
    return f"{original_repository}@{digest}", digest
