"""mkdocs-gallery configuration - produces SVG vector images."""
from functools import partial
from mkdocs_gallery.scrapers import matplotlib_scraper

conf = {
    "image_scrapers": (partial(matplotlib_scraper, format="svg"),),
    "remove_config_comments": True,
    # Suppress plain string repr output (e.g. module docstrings evaluating to
    # a visible string in the first notebook cell).  Matplotlib images are
    # captured by the scraper above, not by repr, so plots are unaffected.
    "capture_repr": (),
    "ignore_repr_types": r"<class 'str'>",
}
